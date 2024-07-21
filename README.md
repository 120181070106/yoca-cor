```相对的改进
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
#---------------------------(voc_annotation.py)------------------------#
annotation_mode     = 2 #基本的集合已被划分于ImageSeg文件夹，现只需生成2007_train.txt、2007_val.txt的目标信息即可（原为0）
#-----------------------------------(utils_map.py)------------------------------#
# 第241和第609行，均加入".manager"变为fig.canvas.manager.set_window_title
    fig.canvas.manager.set_window_title(window_title)#第241行
                fig.canvas.manager.set_window_title('AP ' + class_name)#609行
```
后修正部分（照autodl和featurize调出的默认参数均为1.2和0.6）
```
#---------------------------(utils_bbox.py，双双修正)---------------------------#
# TORCH_1_10 = check_version():随即插入如下↓
def 交(box1, box2, threhold=0.5): # 见信度修正
def iou(当,a=2): # 见信度修正
def bbox_iou(box1, box2, eps=1e-7): # Returns IoU of box1(1,4) to box2(n,4)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    return inter/(w1 * h1 + w2 * h2 - inter + eps)
def 权框(权,框):return torch.sum(权.unsqueeze(-1)*框,0,False)/(torch.sum(权)+1e-9)
#插入后再在第71行替换原函数
    def decode_box(self, inputs): #此函数添加未注释部分替换原"y=torch.cat(())"那句
        # dbox=dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides 
        b = dbox.shape[0] # [b,4,8400]→[b,4,80,80]&[b,4,40,40]&[b,4,20,20]
        低框    = dbox[:,:,:6400].view(b,4,80,80)
        低信    = cls[:,:,:6400].sigmoid().view(b,-1,80,80)
        出信=cls.sigmoid()
        类权 = iou(低框)
        实权 = 1.5+(类权)#1.5→0.5/0.2(随权重的精细化)，featurize上最佳取值为1.2
        低信 *= 实权
        出信[:,:,:6400]=低信.view(b,-1,6400)
        y = torch.cat((dbox, 出信), 1).permute(0, 2, 1)#cls.sigmoid()亦更为出信
#插入后再在第199行又插入↓
            准框,信框,交度阈值=output[i],detections,0.6#准框[nb,6(4位+1信+1类)]
            for d in range(准框.shape[0]):#对全类nms后的准框逐一修正框位
                与各信交度=bbox_iou(准框[d,:4].unsqueeze(0),信框[:,:4],False).squeeze(-1)#[nx,]
                同目框=信框[与各信交度>交度阈值]; 信度集=同目框[:,4]
                叠度=与各信交度[与各信交度>交度阈值]
                准框[d,:4]=权框(信度集*叠度,同目框)[:4]
            output[i] = 准框
#---------------------------(utils_bbox.py，信度修正)---------------------------#
# TORCH_1_10 = check_version():随即插入如下↓
def 交(box1, box2, threhold=0.5):#两(b,w,h,4)→(b,w,h)
    x1=torch.max(box1[...,0],box2[...,0]); y1=torch.max(box1[...,1],box2[...,1])
    x2=torch.min(box1[...,2],box2[...,2]); y2=torch.min(box1[...,3],box2[...,3])
    intersection=torch.clamp(x2-x1,min=0)*torch.clamp(y2-y1,min=0)
    area1=(box1[...,2]-box1[...,0])*(box1[...,3]-box1[...,1])
    area2=(box2[...,2]-box2[...,0])*(box2[...,3]-box2[...,1])
    return intersection / (area1 + area2 - intersection)#改0.001
def iou(当,a=2):#假设是(b,w,h,4),h和w顺序不要紧(b,4,w,h)→(b,w,h,4)→(b,w,h)
    if 当.dim()==3:当=当.unsqueeze(0);a=1;l=0.01#修正输入为(b,w,h,4[原次]),先补批数
    if 当.shape[1]==4 and 当.shape[3]!=4:当=当.transpose(1,2).transpose(2,3)
    左,右,上,下,左上,右下,右上,左下 = 当*0,当*0,当*0,当*0,当*0,当*0,当*0,当*0
    左[:,1:,:,:]=当[:,:-1,:,:]; 右[:,:-1,:,:]=当[:,1:,:,:]#坐标置网宽高后↑↑
    上[:,:,1:,:]=当[:,:,:-1,:]; 下[:,:,:-1,:]=当[:,:,1:,:]
    return 交(当,左)*交(当,右)*交(当,上)*交(当,下)
#插入后再在第61行替换原函数
    def decode_box(self, inputs): #此函数添加未注释部分替换原"y=torch.cat(())"那句
        # dbox, cls, origin_cls, anchors, strides = inputs
        # dbox=dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides 
        b = dbox.shape[0] # [b,4,8400]→[b,4,80,80]&[b,4,40,40]&[b,4,20,20]
        低框    = dbox[:,:,:6400].view(b,4,80,80)
        低信    = cls[:,:,:6400].sigmoid().view(b,-1,80,80)
        出信=cls.sigmoid()
        类权 = iou(低框)
        实权 = 1.2+(类权)#1.5→0.5/0.2(随权重的精细化)，featurize上最佳取值为1.2
        低信 *= 实权
        出信[:,:,:6400]=低信.view(b,-1,6400)
        y = torch.cat((dbox, 出信), 1).permute(0, 2, 1)#cls.sigmoid()亦更为出信
        # y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        # return y
#---------------------------(utils_bbox.py，框位修正)---------------------------#
# TORCH_1_10 = check_version():随即插入如下↓
def bbox_iou(box1, box2, eps=1e-7): # Returns IoU of box1(1,4) to box2(n,4)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    return inter/(w1 * h1 + w2 * h2 - inter + eps)
def 权框(权,框):return torch.sum(权.unsqueeze(-1)*框,0,False)/(torch.sum(权)+1e-9)
#插入后再在第202行又插入↓
                # output[i] = max_detections                       #信框首则为nx↓
            准框,信框,交度阈值=output[i],detections,0.6#准框[nb,6(4位+1信+1类)],
            for d in range(准框.shape[0]):#对全类nms后的准框逐一修正框位
                与各信交度=bbox_iou(准框[d,:4].unsqueeze(0),信框[:,:4],False).squeeze(-1)#[nx,]
                同目框=信框[与各信交度>交度阈值]; 信度集=同目框[:,4]
                叠度=与各信交度[与各信交度>交度阈值]
                准框[d,:4]=权框(信度集*叠度,同目框)[:4]
            output[i] = 准框
            #若需叠入关系框的代码，首维两句的output[i],detections更为"出"和"置信预测"
#-------------------------------(map.ipynb)---------------------------#
    if map_mode == 4: #如已经"pip install pycocotools"下载过coco工具箱，则此倒数第四个有效行注释掉，以便输出本地的各类结果后随即按coco方式得到各评价指标值
#-------------------------(yolo.py，用于显示)---------------------------#
        "model_path"        : 'model_data/b基础633.pth',
        "model_path"        : 'k.pth',#可再换成此自训的不高不低的权值，前权过精难提
```