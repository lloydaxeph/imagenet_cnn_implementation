import argparse
import torch
from torch.utils.data import DataLoader

import config
from model import CustomCNNModel
from custom_dataset import MiniImagenetDataset


def create_test_dataloader(data_path: str, batch_size: int):
    mini_imagenet_ds_test = MiniImagenetDataset(data_path=data_path, transform=config.transform)
    assert batch_size <= len(mini_imagenet_ds_test)
    test_loader = DataLoader(mini_imagenet_ds_test, batch_size=batch_size, shuffle=config.shuffle)
    return mini_imagenet_ds_test,test_loader


def run_test():
    parser = argparse.ArgumentParser(description='CNN Model Test.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=config.imagenet_data_dir_test)
    parser.add_argument('--num_images', type=int, default=config.batch_size)
    parser.add_argument('--print', type=bool, default=False)
    args = parser.parse_args()

    model = CustomCNNModel(n_classes=config.n_classes).to(config.device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    mini_imagenet_ds_test, test_loader = create_test_dataloader(data_path=args.data_path, batch_size=args.num_images)
    c_map = mini_imagenet_ds_test.class_mapping
    score = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            out = model(images.to(config.device))
            for i in range(args.num_images):
                pred_idx = torch.argmax(out[i]).item()
                confidence = out[i][pred_idx]
                predicted = c_map[pred_idx]
                gt = c_map[int(labels[i])]
                if predicted == gt:
                    score += 1
                if args.print:
                    print(f'Predicted: {predicted} - {confidence} | GT: {gt}')
            break
    print(f'Total score: {score}/{args.num_images} | {float(score/args.num_images):.2f}')


if __name__ == '__main__':
    run_test()