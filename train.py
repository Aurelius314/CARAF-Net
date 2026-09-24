import os
import time
from model import *
import numpy as np
from collections import defaultdict
import random
from test import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def trainCARAF(data_loader_dict, optimizer_config, cuda, args, iteration, one_subject):
    source_loaders = data_loader_dict['source_loader']
    test_loader = data_loader_dict['test_loader']
    
    device = torch.device('cuda' if cuda else 'cpu')
    
    log_dir = "log2s"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_acc_{args.dataset_name}_subject_{one_subject}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"主体 {one_subject} 的测试准确率记录\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"格式: [Epoch] 准确率(%)\n")
        f.write("-" * 30 + "\n")
    
    preTrainModel = CARAF_NetModel(
        cuda,
        number_of_source=len(source_loaders),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps
    ).to(device)
    
    optimizer_PreTraining = torch.optim.Adam(
        preTrainModel.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config.get('weight_decay', 1e-5)
    )
    
    all_source_data = []
    all_source_label = []
    
    for source_loader in source_loaders:
        source_data_list = []
        source_label_list = []
        for data, label in source_loader:
            source_data_list.append(data)
            source_label_list.append(label)
        
        if source_data_list:
            all_source_data.append(torch.cat(source_data_list, dim=0))
            all_source_label.append(torch.cat(source_label_list, dim=0))
    
    all_source_data = torch.cat(all_source_data, dim=0)
    all_source_label = torch.cat(all_source_label, dim=0)
    
    print(f"总源域数据大小: {all_source_data.size(0)}")
    
    best_acc = 0
    best_model_state = None
    best_epoch = -1
    
    eval_interval = 1
    
    for epoch in range(args.epoch_preTraining):
        start_time_pretrain = time.time()
        
        preTrainModel.train()
        epoch_losses = []
        
        test_iter = iter(test_loader)
        total_batches = len(test_loader)
        
        for i in range(total_batches):
            try:
                target_data, target_label = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                target_data, target_label = next(test_iter)
            
            p = float(i + epoch * total_batches) / args.epoch_preTraining / total_batches
            m = 2. / (1. + np.exp(-3 * p)) - 1
            
            idx = random.sample(range(all_source_data.size(0)), args.batch_size)
            corres_data = all_source_data[idx]
            corres_label = all_source_label[idx]
            
            if cuda:
                target_data = target_data.to(device)
                target_label = target_label.to(device)
                corres_data = corres_data.to(device)
                corres_label = corres_label.to(device)
            
            lambda_cls = 1.0
            lambda_dom = 0.6
            lambda_dom_I = 0.2
            lambda_pcls = 0.3
            lambda_l = 0.1
            lambda_f = 0.3
            pseudo_label_temp = 1
            pseudo_label_threshold = 0.7
            warmup_epochs = 30
            
            optimizer_PreTraining.zero_grad()
            
            loss_dict = preTrainModel(
                        target_data,
                        corres_data,
                        corres_label,
                        m=m,
                        current_epoch=epoch,
                        total_epochs=100,
                        warmup_epochs=warmup_epochs,
                        lambda_cls=lambda_cls,
                        lambda_dom=lambda_dom,
                        lambda_dom_I=lambda_dom_I,
                        lambda_pcls=lambda_pcls,
                        lambda_l=lambda_l,
                        lambda_f=lambda_f,
                        lambda_entropy=1e-3, 
                        pseudo_label_threshold=pseudo_label_threshold
                    )
            
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(preTrainModel.parameters(), max_norm=1.0)
            
            optimizer_PreTraining.step()
            
            epoch_losses.append(total_loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print(f"Epoch {epoch+1}/{args.epoch_preTraining}, 平均损失: {avg_loss:.4f}, 用时: {pretrain_epoch_time:.2f}秒")
        
        if (epoch + 1) % eval_interval == 0 or epoch == args.epoch_preTraining - 1:
            test_acc = testCARAF(preTrainModel, test_loader, cuda, args)
            
            with open(log_file, 'a') as f:
                f.write(f"[Epoch {epoch+1}] {test_acc:.2f}%\n")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(preTrainModel.state_dict())
                
                model_path = f"best_model_{args.dataset_name}_subject_{one_subject}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': preTrainModel.state_dict(),
                    'optimizer_state_dict': optimizer_PreTraining.state_dict(),
                    'accuracy': best_acc,
                }, model_path)

    
    if best_model_state is not None:
        preTrainModel.load_state_dict(best_model_state)

        with open(log_file, 'a') as f:
            f.write("-" * 30 + "\n")
            f.write(f"最终最佳准确率: {best_acc:.2f}% (Epoch {best_epoch+1})\n")
    
    return preTrainModel, best_acc