import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random

torch.backends.cudnn.enabled = False

#변수 선언 및 정의
total_epoch = 10
Leaning_Rate = 0.001
batch_size = 12
num_classes = 5

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#커스텀 CNN 모델 정의
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        #Conv2d연산
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)  # Softmax 적용
        return out

# 학습 및 평가 함수 정의
def fit(epoch, model, data_loader, phase = 'trainig'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    # 현재까지 학습의 loss와 정확도
    running_loss = 0.0
    running_correct = 0

    #데이터 로더 순회
    for batch_idx, (data, target) in enumerate(data_loader):
        #각 배치 인덱스의 data 텐서와 target 텐서의 device설정
        data, target = data.to(device), target.to(device)

        #학습이면 옵티마이저를 0으로 초기화
        if phase == 'traing':
            optimizer.zero_grad()

        #각 배치 인덱스에 따른 결과값 각 이미지별 종류와의 유사값
        output = model(data)

        # 옵티마이저 정의 및 설정
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss =  F.cross_entropy(output, target)
        running_loss += F.cross_entropy(output, target, reduction='sum').data

        # 학습이면 역전파를 통해 가중치 업데이트
        if phase == 'training':
            loss.backward()
            optimizer.step()

        # 내가 학습을 통해 예측한 예측값
        preds = output.data.max(dim=1, keepdim=True)[1]

        #현재까지의 예측해서 맞추 갯수를 계산
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

    #최종 loss와 정확도 계산
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)

    #최종 loss랑 정확도 반화
    return loss, accuracy

#학습 함수
def train():
    # 데이터 셋의 트랜스폼 정의
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 데이터 셋 선언
    train = ImageFolder(root="./data/Animals/train", transform=transform)


    # 데이터로더를 생성합니다.
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)


    #데이터 불러오기 끝났음을 알림
    print("------------- data load finished -------------------------")

    # 모델 인스턴스화
    model = CNNModel(num_classes)
    model = model.to(device)
    epoch_min_loss = 10
    for epoch in range(1, total_epoch + 1):

        print("-----------training: {} epoch-----------".format(epoch))

        #학습 한번, 테스트 한번 실행
        epoch_loss, epoch_accuracy = fit(epoch, model, train_dataloader, phase='training')

        print(f'Epoch {epoch}/{total_epoch} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

        #10번의 에폭 마다 모델 저장
        if epoch_loss <= epoch_min_loss:
            epoch_min_loss = epoch_loss
            savePath = "./model/model_Animals.pth"
            torch.save(model.state_dict(), savePath)
            print("file save at {}".format(savePath))
    # 마지막 에폭 모델
    savePath = "./model/model_Animals_Final.pth"
    torch.save(model.state_dict(), savePath)
    print("file save at {}".format(savePath))

def predict_image(img_path, model, class_list, transform):
    #이미지 하나 불러오기
    img = Image.open(img_path)
    #1차원 배열 바꾸는거 같은데
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        #불러오 이미지 배열을 텐서로
        outputs = model(img)
        #outputs의 데이터값이 이상하게 나오고 있는데 정확한 이유를 찾지 못함
        #학습이 제대로 안된건지 아예 잘못된 구조인거인지 알수 없음  모든 출력이 0번째가 제일 큰값을 출력한다
        #print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted.item())
        return class_list[predicted.item()]


def imshow(img_path, cmap=None):
    """Imshow for Tensor."""
    # 이미지 하나 불러오기
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    #불러온 이미지를 numpy배열로 만듬
    inp = img.cpu().detach().numpy()
    inp = np.squeeze(inp, axis=0)   #이미지를 불러올때 맨앞 배치 1까지 불러와서 첫번째 차원을 없애야함
    inp = inp.transpose((1, 2, 0))

    #이미지 역 정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    plt.show()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images.append(img_path)  # 이미지 파일 경로 저장
    return images

# 랜덤하게 하나의 이미지를 선택하는 함수
def get_random_image(images):
    return random.choice(images)



if __name__ == '__main__':
    classes_list = ['cat', 'chicken', 'cow', 'dog', 'sheep']
    #학습 및 모델 저장
    #train()
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    save_path = './model/model_Animals.pth'
    load_Model = CNNModel(num_classes)
    load_Model.load_state_dict(torch.load(save_path))
    load_Model.to(device)
    load_Model.eval()

    image_folder_path = './data/Animals/Final_image'
    images = load_images_from_folder(image_folder_path)
    random_image_path = get_random_image(images)

    predict_class = predict_image(random_image_path, load_Model, classes_list, transform = transform)
    print(predict_class)

    imshow(random_image_path, predict_class)
    print("End")
