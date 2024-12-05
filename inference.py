import torch
import argparse
import tinycudann as tcnn
from manage_objs import *
from sdf.utils2 import *
from hash_imports.encodings import HashEncoding

def save_mesh(models, mode,no_of_rots,W=None,N=None,M=None,save_path='inferences/', resolution=256):
        for j, model in enumerate(models):
            if N == None and M == None and W==None:
                save_path1 = save_path+f'{j}.obj'
            elif mode == 'lora': 
                save_path1 = save_path+f'N_M_{j}.obj'
            elif mode == 'normal': 
                save_path1 = save_path+f'W_{j}.obj'
            print(f"==> Saving mesh to {save_path1}")
            if no_of_rots == 1:
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
                print(f"==> Finished saving mesh {j+1}/{len(models)}.")
            # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
            # mesh.export(save_path)
            else:
                w =W
                
                for i in range(no_of_rots):
                    save_path2 = save_path +f'obj_{j}_rot_{i}.obj'
                    os.makedirs(os.path.dirname(save_path2), exist_ok=True)

                    def query_func(pts,model):
                        pts = pts.to('cuda')
                        with torch.no_grad():   
                            with torch.cuda.amp.autocast(enabled=True):
                                # model.encoder.params = torch.nn.Parameter(model.encoder.params*w.params)
                                sdfs = model(pts,mode,w,N,M) #TODO
                        return sdfs

                    bounds_min = torch.FloatTensor([-1, -1, -1])
                    bounds_max = torch.FloatTensor([1, 1, 1])
                    
                    vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func, model=model)
                    mesh = trimesh.Trimesh(vertices, triangles, process=False)
                    mesh.export(save_path2)
                    w = torch.matmul(w,W)
                    print(f"==> Finished saving mesh {j}/{len(models)}.")
                    

        print(f"==> Finished saving meshes.")


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="Process two integers n and m.")

    # Add arguments for n and m
    # parser.add_argument("n", type=int, help="An integer value for n")
    # parser.add_argument("d", type=int, help="An integer value for m")

    # Parse arguments
    # args = parser.parse_args()

    # # Access the arguments
    # n = args.n
    # d = args.d

    mode = 'normal'
    W = None
    N = None
    M = None 
    B1 = None 
    B2 = None
    no_of_rots = 1
    # obj_path_pre = 'GT_objs/'
    workspace1 = f'hash_workspace_obj12_7_8_10/'
    _n = 2
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
    # tcnn_network = torch.load(workspace1+'GT_mlp_7_8_12.pth')
    models = []
    for idx in range(_n):
        model = SDFNetwork( tcnn_network, encoding="hashgrid")
        # enc_state_dict = torch.load(f'saved_models/enc_{idx}.pth')
        # model.encoder = torch.load(workspace1+f'/GT_enc{idx+10}_{n}_{d}_12.pth')
        # model.encoder = torch.load('hash_workspace_obj12_7_8_10/GT_enc.pth')
        # model.backbone = torch.load("hash_workspace_obj12_7_8_10/GT_mlp.pth")
        model.encoder = torch.load(f'hash_workspace_obj12_7_8_10/GT_enc_{idx}.pth')
        # model.encoder.hash_table = torch.nn.Parameter(torch.load('new_hash_table.pth'))
        model.backbone = torch.load("hash_workspace_obj12_7_8_10/GT_mlp.pth")
        models.append(model)
    # model2 = SDFNetwork(tcnn_network, encoding="hashgrid") 
    # models = [model, model2]
    # print(models)
    # models[0].encoder = torch.load('rotated_enc.pth')
    # models[0].backbone = torch.load("rotated_mlp.pth")
    # models[0].encoder=togarv
    # rch.load('GT_enc2.pth')
    # models[0].backbone=torch.load('GT_mlp2.pth')
    
    # dim = models[0].encoder.hash_table.shape[0]
    
    # N = torch.load('hash_workspace_obj12_7_8_10/N.pth')
    # M = torch.load('hash_workspace_obj12_7_8_10/M.pth')
    # w = torch.load('w.pth')
    W= torch.load(workspace1+'W.pth')
    # W = torch.matmul(W,W)# W2 = 2*W
    
    #trainer.save_mesh(os.path.join(workspace1, 'results', 'output.ply'), 1024)
    save_mesh(models,mode,no_of_rots,W,N,M,save_path='inferences/GT/')     
