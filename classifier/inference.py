import timm
import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from utils import Resize_with_Ratio, MyImageFolder, Cutout
from PIL import Image

mean, std = (0.42760366, 0.4122089, 0.38598374), (0.17665266, 0.17428997, 0.16621284)

classes = ['中型坦克', '其他', '军用卡车', '军用吉普', '军用皮卡', '加油车', '工程车', '机库', '民用车', '油库', '装甲车', '轻型坦克', '重型坦克', '铲雪车', '雷达']


def inference(model_name, num_classes, snapshot_path, image_path, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(snapshot_path, map_location=torch.device('cpu')))
    model.to(torch.device(device))

    test_augs = transforms.Compose([
        Resize_with_Ratio((300, 300)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img = Image.open(image_path).convert("RGB")
    img = test_augs(img)

    with torch.no_grad():
        model.eval()  # 评估模式
        img = img.unsqueeze(dim=0)

        out = model(img.to(device))
        out = F.softmax(out, dim=1)
        print("out: ", out)
        ship_class = classes[out.argmax(dim=1).tolist()[0]]
        print("ship_class: ", ship_class)


if __name__ == '__main__':
    model_name = "seresnext26t_32x4d"
    num_classes = 15
    snapshot_path = './output/2023_06_26_14_37_08_seresnext26t_32x4d/snapshot_134.pth'
    image_path = '/home/xlz/Desktop/project/地面装备/UI/server/crop/1693470430.0278566.jpg'
    device = 'cuda'
    inference(model_name, num_classes, snapshot_path, image_path, device)
