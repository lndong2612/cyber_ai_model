from PIL import Image
import torch
import torchvision.transforms as transforms
from app.neural_model.mobile_facenet.utils import extract_face, create_model_mobileface


def predict_image(model, label_object, img, device):
    model.eval()
    transform_norm = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Resize((112,112)),
            lambda x: x*255
        ]
        )
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    # input = Variable(image_tensor)

    with torch.no_grad():
        output = model(img_normalized)
        probs = torch.nn.functional.softmax(output, dim=1)
        score, predicted = torch.max(probs, 1)
        return (label_object[predicted.item()], score)

if __name__ == "__main__":
    path_weight = "data_crop/weights/best.pt"
    device = "cpu"
    model = create_model_mobileface(device, 5, path_weight)
    label_object = ["Bien", "Cuong", "LeDong", "Phu", "Vu"]
    img = Image.open("data_crop/Phu/frame206302021-090911.jpg")
    predict_image(model, label_object, img, device)

