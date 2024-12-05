import torch
import argparse
import tinycudann as tcnn
from manage_objs import *
from sdf.utils import *
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default = "torch-ngp/OmniObj3D") #path  to recusively check for obj's
    #parser.add_argument('path1', type=str)
    #parser.add_argument('path2', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace1', type=str, default='workspace')
    #parser.add_argument('--workspace2', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    opt = parser.parse_args()
    print(opt)
    directory_path = opt.path
    obj_files = find_obj_files(directory_path)
    cat_dict = get_category_dict(obj_files)
    paths = get_final_objs(cat_dict, 10,2)
    print()
    print(paths)
    print()
    #copy_files_with_index(paths,'GT_objs')
    #paths = []
    #paths.append(opt.path1)
    #paths.append(opt.path2)
    seed_everything(opt.seed)
    n = len(paths)
    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.netowrk_ff import SDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn import SDFNetwork        
    else:
        from sdf.netowrk import SDFNetwork

    
    tcnn_network = tcnn.Network(
                                        n_input_dims=32,
                                        n_output_dims=1,
                                        network_config={
                                            "otype": "FullyFusedMLP",
                                            "activation": "ReLU",
                                            "output_activation": "None",
                                            "n_neurons": 64,
                                            "n_hidden_layers": 2,
                                        },
                                    )
    
    models = []
    for idx in range(n):
        model = SDFNetwork( tcnn_network, encoding="hashgrid")
        models.append(model)
    #model2 = SDFNetwork(tcnn_network, encoding="hashgrid") 
    #models = [model, model2]
    #print(models)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

    else:
        from sdf.provider import SDFDataset
        from loss import mape_loss
        train_datasets = []
        train_loaders = []
        valid_datasets = []
        valid_loaders = []
        for i in range(n):
            train_dataset = SDFDataset(paths[i], size=20, num_samples=2**18)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            train_datasets.append(train_dataset)
            train_loaders.append(train_loader)

        for i in range(n):  
            valid_dataset = SDFDataset(paths[i], size=1, num_samples=2**18) # just a dummy
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
            valid_datasets.append(valid_dataset)
            valid_loaders.append(train_loader)

        criterion = mape_loss # torch.nn.L1Loss()

        enc_optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()}
            #{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        net_optimizer = lambda model: torch.optim.Adam([
            #{'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6}
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)


        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        subset_n = 10
        # for _ in range(1000):
        #     rand_idx = torch.randint(0, n, (subset_n,)).tolist()
        #     subset_models = [models[idx] for idx in rand_idx] 
        #     subset_train_loaders = [train_loaders[idx] for idx in rand_idx]
        #     subset_valid_loaders = [valid_loaders[idx] for idx in rand_idx]
        for i in range(0, n, subset_n):
    # Select the subset of models and loaders in successive groups of 10
            subset_models = models[i:i + subset_n]
            subset_train_loaders = train_loaders[i:i + subset_n]
            subset_valid_loaders = valid_loaders[i:i + subset_n]
            trainer = Trainer('ngp', subset_n, models, workspace=opt.workspace1, enc_optimizer=enc_optimizer,  net_optimizer=net_optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1)
            trainer.train(subset_train_loaders, subset_valid_loaders,5)

        # also test
        trainer.save_mesh(os.path.join(opt.workspace1, 'results', 'output.ply'), 1024)   
