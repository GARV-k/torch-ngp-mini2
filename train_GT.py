import torch
import argparse
import tinycudann as tcnn
from manage_objs import *
from sdf.utils2 import *
from hash_imports.encodings import HashEncoding

def save_mesh(models, w,save_path='meshes_using_hashencoder/', resolution=256):
        for j, model in enumerate(models):
            save_path1 = save_path+f'{j+10}.obj'
            print(f"==> Saving mesh to {save_path1}")

            os.makedirs(os.path.dirname(save_path1), exist_ok=True)

            def query_func(pts,model):
                pts = pts.to('cuda')
                with torch.no_grad():   
                    with torch.cuda.amp.autocast(enabled=True):
                        # model.encoder.params = torch.nn.Parameter(model.encoder.params*w.params)
                        sdfs = model(pts) #TODO
                return sdfs

            bounds_min = torch.FloatTensor([-1, -1, -1])
            bounds_max = torch.FloatTensor([1, 1, 1])
            
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func, model=model)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(save_path1)
        # vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
            print(f"==> Finished saving mesh {j+1}/{len(models)}.")
        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(save_path)

        print(f"==> Finished saving meshes.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process two integers n and m.")

    # Add arguments for n and m
    parser.add_argument("n", type=int, help="An integer value for n")
    parser.add_argument("d", type=int, help="An integer value for m")

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    n = args.n
    d = args.d

    
    obj_path_pre = 'GT_objs/'
    workspace1 = f'hash_workspace_10objs_{n}_{d}_12dim/'
    _n = 10
    from sdf.network_tcnn import SDFNetwork            
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
    
    
    # mlp_state_dict = torch.load('saved_models/mlp.pth')
    # tcnn_network.load_state_dict(mlp_state_dict)
    tcnn_network = torch.load(workspace1+'GT_mlp_7_8_12.pth')
    models = []
    for idx in range(10):
        model = SDFNetwork( tcnn_network, encoding="hashgrid")
        # enc_state_dict = torch.load(f'saved_models/enc_{idx}.pth')
        model.encoder = torch.load(workspace1+f'/GT_enc{idx+10}_{n}_{d}_12.pth')
        models.append(model)
    # model2 = SDFNetwork(tcnn_network, encoding="hashgrid") 
    # models = [model, model2]
    # print(models)
    # models[0].encoder = torch.load('rotated_enc.pth')
    # models[0].backbone = torch.load("rotated_mlp.pth")
    # models[0].encoder=togarv
    # rch.load('GT_enc2.pth')
    # models[0].backbone=torch.load('GT_mlp2.pth')
    
    dim = models[0].encoder.hash_table.shape[0]
    
    w = nn.Parameter(torch.randn(2, 2, device='cpu', requires_grad=True))
    # w = torch.load('w.pth')
    
    lr = 5e-6

    from sdf.provider import SDFDataset
    from loss import mape_loss
    train_datasets = []
    train_loaders = []
    valid_datasets = []
    valid_loaders = []
    # for i in range(_n):
    for i in range(10,20):
        train_dataset = SDFDataset(obj_path_pre+f'{i}.obj', size=100, num_samples=2**18,sample_ratio_n=n,sample_ratio_d=d)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_datasets.append(train_dataset)
        train_loaders.append(train_loader)

    # for i in range(_n):  
    for i in range(10,20):
        valid_dataset = SDFDataset(obj_path_pre+f'{i}.obj', size=1, num_samples=2**18,sample_ratio_n=n,sample_ratio_d=d) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
        valid_datasets.append(valid_dataset)
        valid_loaders.append(train_loader)

    criterion = mape_loss # torch.nn.L1Loss()

    enc_optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': model.encoder.parameters()}
        #{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    net_optimizer = lambda model: torch.optim.Adam([
        #{'name': 'encoding', 'params': model.encoder.parameters()},
        {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6}
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    w_optimizer = lambda w: torch.optim.Adam([
        {'name': 'encoding', 'params': [w]}
        #{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    subset_n = 10
    trainer = Trainer('ngp', _n, models,w, workspace=workspace1, enc_optimizer=enc_optimizer,  net_optimizer=net_optimizer, w_optimizer = w_optimizer, criterion=criterion, ema_decay=0.95, fp16=False, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1,scheduler_update_every_step=False)
    trainer.train(train_loaders, valid_loaders,40,subset_n)
    # w_state_dict = w.state_dict()
    # path = f"saved_models/w.pth"
    # torch.save(w_state_dict, path)
    
    
    for idx,model in enumerate(models):
        torch.save(model.encoder,workspace1+f'/GT_enc{idx+10}_{n}_{d}_12.pth')
        torch.save(model.backbone,workspace1+f'/GT_mlp_{n}_{d}_12.pth')
    
    #     # w_state_dict = w.state_dict()
    #     # path = f"saved_models/w.pth"    
    #     # torch.save(w_state_dict, path)
    #     torch.save(model.encoder,'shifted_enc .pth')
    # torch.save(models[0].encoder,workspace1+f'/GT_enc{10+trial}_{n}_{d}_12.pth')
    # torch.save(models[0].backbone,workspace1+f'/GT_mlp_{n}_{d}_12.pth')
    # torch.save(w,'w.pth')
    
    print("Completed saving path of models")
    
    
    #trainer.save_mesh(os.path.join(workspace1, 'results', 'output.ply'), 1024)
    save_mesh(models,w,save_path=workspace1+f'hash_GT_objs_{n}_{d}_12/')
        
