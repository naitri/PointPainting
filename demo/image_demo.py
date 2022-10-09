from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette,get_classes


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

     PATH = "ford_av/mav0/cam0/data/"
    Copy_to_path="cam0/"
    counter = 0
    total = (len([name for name in os.listdir(PATH)]))
    objects = ["car","bicycle","rider","person","motorcycle"]
    for filename in sorted(os.listdir(PATH)):
        print('Percentage Done :',(counter/total)*100, end='\r')
        img = cv2.imread(os.path.join(PATH, filename)) 
        
    # test a single image
        result = inference_segmentor(model, args.img)
        seg_img = result[0]
        palette = get_palette(args.palette)
        classes = get_classes(agrs.palette)
        mask = np.zeros(semantics.shape[0])
       
        for label,objt in enumerate(zip(palette,classes)):

            name = objt[1]
            color = objt[0]
           
            if name in objects:
                idx = np.where(seg_img == label)[0]
                mask[idx] = 255
                    
            else:
                idx = np.where(seg_img == label)[0]
                mask[idx] = 0
         

    


if __name__ == '__main__':
    main()
