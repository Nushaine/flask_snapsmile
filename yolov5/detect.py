import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt

from flask import jsonify

# used to lower brightness in images


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # create directory to write confidence scores and bbox coords to
    directory = "data"
    parent_dir = "../yolov5/inference/"
    data_path = os.path.join(parent_dir, directory)
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)

    # create lists to store confidence and coordinates
    confidences = []
    scores = []

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    imgsz = check_img_size(
        imgsz, s=model.model[-1].stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # lists to store detections per class
    empty_lists = [[] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    true_false_list = [False, False, False, False]
    image = 0
    for path, img, im0s, vid_cap in dataset:
        image += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' %
                                                        dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            print("DET", det)

            # make copies of images depending on # of classes
            image_copies = [im0.copy() for _ in range(len(names))]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    print("N", n)
                    print('det', det[:, -1])
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    print("S", s)

                # create new file to write conf scores and bbox coords to
                file_name = 'data.txt'
                file_path = os.path.join(data_path, file_name)

                bbox = 0
                plot_evolution_results

                # Write results
                for *xyxy, conf, cls in det:
                    bbox += 1
                    coords_per_box = []
                    print("CLS: ", int(cls))
                    true_false_list[int(cls)] = True
                    for iter in range(0, len([*xyxy])):
                        # get all coords per box
                        coords_per_box.append(([*xyxy][iter].item()))
                    # append to list for all box coords
                    # each list represents a different class
                    empty_lists[int(cls)].append([coords_per_box, conf.item()])
                    confidences.append(
                        [f"IMAGE {image}, BBOX {bbox}: ", coords_per_box])
                    # get confidence scores
                    scores.append(
                        [f"IMAGE {image}, BBOX {bbox}: ", conf.item()])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') %
                                    (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)

            # make images w. detections per class
            for index, lists in enumerate(empty_lists):
                print("************new iter *************")
                label = names[index].capitalize()
                image = image_copies[index]
                if len(lists) == 0:
                    # lower brightness in image
                    image = change_brightness(image, value=-80)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    text = "No " + label + " found"

                    # get boundary of this text
                    textsize = cv2.getTextSize(text, font, 1, 2)[0]

                    # get coords based on boundary
                    textX = (image.shape[1] - textsize[0]) // 2
                    textY = (image.shape[0] + textsize[1]) // 2

                    # add text centered on image
                    cv2.putText(image, text, (textX, textY),
                                font, 1, (255, 255, 255), 2)

                else:
                    # draw boxes on images
                    for detections in lists:
                        print("BBOX INDEX", int(lists.index(detections)))
                        print("LIST @ BBOX INDEX",
                              lists[int(lists.index(detections))])
                        print("DETECTIONS", lists[int(
                            lists.index(detections))][0])
                        plot_one_box(lists[int(lists.index(detections))][0], image, color=colors[int(
                            cls)], line_thickness=5)
                        plt.imshow(image)

                # make path for image -> folder name is based on image class
                new_path = str(os.getcwd()) + \
                    "/yolov5/inference/output/" + label
                print("new path: " + new_path)
                # new_path = "D:/work/snapsmile/computer visoin/flask/yolov5/inference/output/" + label
                print("names", label)

                # delete folder if already exists
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)  # delete output folder
                os.makedirs(new_path)  # make new output folder
                cv2.imwrite(os.path.join(new_path, 'detections.jpg'), image)

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    with open(os.path.join(data_path, "data.txt"), 'w') as d:
        d.write("confidences" + "\n")
        for item in confidences:
            d.write(str(item) + "\n")
        d.write("\n" + "\n")
        d.write("scores" + "\n")
        for item in scores:
            d.write(str(item) + "\n")

    print("TRUE: ", true_false_list)
    file_name2 = 'true_false.txt'
    file_path2 = os.path.join(data_path, file_name2)
    with open(file_path2, "w") as f:
        f.write(str(true_false_list))
        f.close()

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+',
                        type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    opt = parser.parse_args()
    print("OPT", opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
