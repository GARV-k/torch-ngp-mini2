
import torch
import argparse
import tinycudann as tcnn
from manage_objs import *
from sdf.utils2 import *
from hash_imports.encodings import HashEncoding

def save_mesh(models, mode,W=None,N=None,M=None,save_path='meshes_using_hashencoder/', resolution=256):
        for j, model in enumerate(models):
            if N == None and M == None and W==None:
                save_path1 = save_path+f'0.obj'
            elif mode == 'lora': 
                save_path1 = save_path+f'N_M_0.obj'
            elif mode == 'normal': 
                save_path1 = save_path+f'W_0.obj'
            print(f"==> Saving mesh to {save_path1}")

            os.makedirs(os.path.dirname(save_path1), exist_ok=True)

            def query_func(pts,model):
                pts = pts.to('cuda')
                with torch.no_grad():   
                    with torch.cuda.amp.autocast(enabled=True):
                        # model.encoder.params = torch.nn.Parameter(model.encoder.params*w.params)
                        sdfs = model(pts,mode,W,N,M) #TODO
                return sdfs

            bounds_min = torch.FloatTensor([-1, -1, -1])
            bounds_max = torch.FloatTensor([1, 1, 1])
            
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func, model=model)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(save_path1)
        # vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
            print(f"==> Finished saving mesh {idx}/{len(models)-1}.")
        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(save_path)

        print(f"==> Finished saving meshes.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process two integers n and m.")

    # Add arguments for n and m
    # parser.add_argument("n", type=int, help="An integer value for n")
    # parser.add_argument("d", type=int, help="An integer value for m")
    # parser.add_argument("hashmap_size", type=int, help="An integer value for log hashmap size")
    # parser.add_argument("obj_no", type=int, help="An integer value for number of obj")

    # # Parse arguments
    # args = parser.parse_args()

    # # Access the arguments
    # n = args.n
    # d = args.d
    # hashmap_size = args.hashmap_size
    # obj_no = args.obj_no
    n = 7
    d = 8
    hashmap_size = 10
    obj_no = 12
    no_of_rotations = 1
    mode = 'normal'
    obj_path_pre = 'datasets/rot_after_inf/'
    # obj_path_pre_list = [obj_path_pre+'/'
    #                      ,obj_path_pre+'_180/'
    #                     #,obj_path_pre+'_270/'
    #                     #  ,obj_path_pre+'_360/'
    #                      ]
    workspace1 = f'hash_workspace_obj{obj_no}_{n}_{d}_{hashmap_size}/'
    _n = 2
    from sdf.network_tcnn import SDFNetwork            
    
    tcnn_network = torch.load(workspace1+'GT_mlp.pth')
    models = []
    for idx in range(_n):
        model = SDFNetwork(tcnn_network, encoding="hashgrid",hashmap_size=hashmap_size)
        # enc_state_dict = torch.load(f'saved_models/enc_{idx}.pth')
        # model.encoder.load_state_dict(enc_state_dict)
        model.encoder = torch.load(workspace1+f'GT_enc_{idx}.pth')
        models.append(model)
        
    W=None
    N=None
    M=None
    B1=None
    B2 = None
    
    dim = models[0].encoder.hash_table.shape[0]	
    print(f'W:{dim}x{dim}')    
    # W = nn.Parameter(torch.randn(dim, dim, device='cuda', requires_grad=True))  #w-commment
    # N = nn.Parameter(torch.randn(dim, dim2, device='cuda', requires_grad=True))
    # M = nn.Parameter(torch.randn(dim2, dim, device='cuda', requires_grad=True))
    
    W = torch.load(workspace1+'W.pth')  
    
    lr = 1e-5

    from sdf.provider import SDFDataset
    from loss import mape_loss
    train_datasets = []
    train_loaders = []
    valid_datasets = []
    valid_loaders = []
    # for i in range(_n):
    # for j in range(no_of_rotations):
    #     for i in [obj_no]:
    #         train_dataset = SDFDataset(obj_path_pre_list[j]+f'{i}.obj', size=100, num_samples=2**14,sample_ratio_n=n,sample_ratio_d=d)
    #         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    #         train_datasets.append(train_dataset)
    #         train_loaders.append(train_loader)
    
    
    for i in range(_n):
        train_dataset = SDFDataset(obj_path_pre+f'{i}.obj', size=100, num_samples=2**18,sample_ratio_n=n,sample_ratio_d=d)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_datasets.append(train_dataset)
        train_loaders.append(train_loader)

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
       

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # scheduler = lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer,[6,6,4,4,2], gamma=0.1, last_epoch=-1)
    subset_n = 2
    trainer = Trainer('ngp', _n, models,W,N,M, B1,workspace=workspace1,enc_optimizer=enc_optimizer,  net_optimizer=net_optimizer, w_optimizer = w_optimizer, criterion=criterion, ema_decay=0.95, fp16=False, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1,scheduler_update_every_step=False)
    trainer.w_train(mode,False,train_loaders,500,subset_n)

    
    # save_mesh(models,mode,W,N,M,save_path=workspace1)
        
