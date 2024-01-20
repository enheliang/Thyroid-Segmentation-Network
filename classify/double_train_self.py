import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from modal_tiqu import resnet50
from double_dataloader_csv import MyDataset
from just_fc import Just_FC




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 # transforms.Resize([224, 224]),
                                 transforms.RandomHorizontalFlip(),    # 随机旋转
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),#keep the w/h ratio,then resize the min-edge to 256
                               transforms.CenterCrop(224),
                               # transforms.Resize([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_transform1 = {
    "train": transforms.Compose([  # transforms.RandomResizedCrop(224),
                                 transforms.Resize([224, 224]),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([# transforms.Resize(256),#keep the w/h ratio,then resize the min-edge to 256
                               # transforms.CenterCrop(224),
                               transforms.Resize([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_csv = ''
val_csv = ''

img_sum_path = ""  #

Cutting_Center = ""

mask = ""  #
Cropped_edges = ""

train_dataset = MyDataset(root=img_sum_path, root1=Cutting_Center, root2=mask, root3=Cropped_edges, csv_dir=train_csv,
                          transform=data_transform["train"], transform1=data_transform1["train"])
train_num = len(train_dataset)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)#when num_workers>0,the speed of transform will so fast

validate_dataset = MyDataset(root=img_sum_path, root1=Cutting_Center, root2=mask, root3=Cropped_edges, csv_dir=val_csv,
                             transform=data_transform["val"], transform1=data_transform1["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

net1 = resnet50(num_classes=2)
print('total number of paramerters in networks is {}'.format(sum(x.numel() for x in net1.parameters())))
#load pretrain weights
model_weight_path1 = ""
pre_weights = torch.load(model_weight_path1)
model_dict = net1.state_dict()
#delete classifier weights
pretrained_dict = {k: v for k, v in pre_weights.items() if (k in model_dict and 'fc' not in k)}
model_dict.update(pretrained_dict)
net1.load_state_dict(model_dict)

net2 = resnet50(num_classes=2)
print('total number of paramerters in networks is {}'.format(sum(x.numel() for x in net2.parameters())))
#load pretrain weights
model_weight_path2 = ""
pre_weights = torch.load(model_weight_path2)
model_dict = net2.state_dict()

pretrained_dict = {k: v for k, v in pre_weights.items() if (k in model_dict and 'fc' not in k)}

model_dict.update(pretrained_dict)
net2.load_state_dict(model_dict)

# net3 = resnet50(num_classes=2)
# print('total number of paramerters in networks is {}'.format(sum(x.numel() for x in net3.parameters())))
# #load pretrain weights
# model_weight_path3 = "" #"./resnet101-5d3b4d8f.pth"
# pre_weights = torch.load(model_weight_path3)
# model_dict = net3.state_dict()

# pretrained_dict = {k: v for k, v in pre_weights.items() if (k in model_dict and 'fc' not in k)}

# model_dict.update(pretrained_dict)
# net3.load_state_dict(model_dict)

net4 = resnet50(num_classes=2)
print('total number of paramerters in networks is {}'.format(sum(x.numel() for x in net4.parameters())))
#load pretrain weights
model_weight_path4 = ""
pre_weights = torch.load(model_weight_path4)
model_dict = net4.state_dict()

pretrained_dict = {k: v for k, v in pre_weights.items() if (k in model_dict and 'fc' not in k)}

model_dict.update(pretrained_dict)
net4.load_state_dict(model_dict)
#
net = Just_FC()
print('total number of paramerters in networks is {}'.format(sum(x.numel() for x in net.parameters())))
# model_weight_path = "./quanzhong40/resNet50_attention_justfc.pth"       # ./save_weights/resNet50_double_justfc.pth
# pre_weights = torch.load(model_weight_path)
# #delete classifier weights
# # pre_dict = {k:v for k, v in pre_weights.items()}#if 'classifier' not in k}# type of weights is dict
# pre_dict =  {k: v for k, v in pre_weights.items() if (k in model_dict and 'fc' not in k)}
# net.load_state_dict(pre_dict, strict=False)

net = net.to(device)
net1 = net1.to(device)
net2 = net2.to(device)
# net3 = net3.to(device)
net4 = net4.to(device)


loss_function = nn.CrossEntropyLoss()   # CrossEntropyLoss() ; BCEWithLogitsLoss()
optimizer = optim.Adam(net1.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=8, verbose=True)

save_path = ''
save_path1 = ''
save_path2 = ''
# save_path3 = ''
save_path4 = ''

epochs = 100
tra_acc_list = []
val_acc_list = []
tra_loss_list = []
val_loss_list = []

best_acc = 0.0
best_specificity = 0.0
best_sensitivity = 0.0

for epoch in range(epochs):
    acc = 0.0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    # train
    net.train()           ##will BN
    net1.train()
    net2.train()
    # net3.train()
    net4.train()
    running_loss = 0.0
    acc_0 = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels, CAM_img, fenge_img, fenge_img2 = data
        optimizer.zero_grad()
        output = net(net1(images.to(device)),
                     net2(CAM_img.to(device)),
                     # net3(fenge_img.to(device)),
                     net4(fenge_img2.to(device))
                     )

        # CrossEntropyLoss()
        loss = loss_function(output, labels.to(device))

        loss.backward()
        optimizer.step()##optimizer.step() is in mi-ni batch,butscheduler.step() is in epoch
        tra_y = torch.max(output, dim=1)[1]
        acc_0 += (tra_y == labels.to(device)).sum().item()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()##will not BN
    net1.eval()
    net2.eval()
    # # net3.eval()
    net4.eval()
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels, CAM_img, fenge_img, fenge_img2 = val_data
            outputs = net(net1(val_images.to(device)),
                          net2(CAM_img.to(device)),
                          # net3(fenge_img.to(device)),
                          net4(fenge_img2.to(device))
                          )# eval model only have last output layer
            # CrossEntropyLoss()
            val_loss = loss_function(outputs, val_labels.to(device))

            predict_y = torch.max(outputs, dim=1)[1]

            # 计算真实负例和假正例的数量
            true_negatives += torch.eq(predict_y, 0).sum().item()
            false_positives += torch.eq(predict_y, 1).sum().item()

            # 计算真实正例和假负例的数量
            true_positives += torch.eq(predict_y, val_labels.to(device)).sum().item()
            false_negatives += torch.eq(predict_y, 1 - val_labels.to(device)).sum().item()

            acc += (predict_y == val_labels.to(device)).sum().item()


            validate_loader.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

        val_accurate = acc / val_num
        specificity = true_negatives / (true_negatives + false_positives)
        sensitivity = true_positives / (true_positives + false_negatives)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            torch.save(net1.state_dict(), save_path1)
            torch.save(net2.state_dict(), save_path2)
            torch.save(net4.state_dict(), save_path4)

        if specificity > best_specificity:
            best_specificity = specificity

        if sensitivity > best_sensitivity:
            best_sensitivity = sensitivity

        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, running_loss / step, val_accurate))
        print('Specificity: %.3f  Sensitivity: %.3f' % (specificity, sensitivity))



        tra_acc_list.append(acc_0 / train_num)
        tra_loss_list.append(running_loss / step)
        val_acc_list.append(val_accurate)
        val_loss_list.append(val_loss)
    scheduler.step(val_accurate)

print('Best Accuracy: %.3f' % best_acc)
print('Best Specificity: %.3f' % best_specificity)
print('Best Sensitivity: %.3f' % best_sensitivity)
print('Finished Training')
