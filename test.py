import argparse
from rivagan import RivaGAN
import pdb
import numpy as np
import torch
import os
import sys
import subprocess
import re

def get_frame_rate(file_path):
    cmd = ['ffprobe', file_path, '-select_streams', 'v', '-show_entries', 'stream=avg_frame_rate']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    status = p.returncode

    num = None
    den = None
    res = re.findall(r'\[STREAM\]\navg_frame_rate=(-?\d+)\/(\d+)\n[\d\D]+', out.decode('utf8'))
    if len(res) > 0:
        num = int(res[0][0])
        den = int(res[0][1])
    return num, den

def get_acc(y_true, y_pred):
    # assert y_true.size() == y_pred.size()
    return (torch.Tensor(y_pred) >= 0.0).eq(torch.Tensor(y_true) >= 0.5).sum().float().item() / torch.Tensor(y_pred).numel()

def encode_video(input_path, output_path, model_path, data, fps=20.0):
    model = RivaGAN.load(model_path)
    model.encode(input_path, data, output_path, fps)
    print('encode successfully!')

def val_decode_video(encoded_path, model_path, data):
    model = RivaGAN.load(model_path)
    acc = []
    for recovered_data in model.decode(encoded_path):
        acc.append(get_acc(data, recovered_data))
        print('now acc:%s' % (acc[-1]))
        print('now averate acc:%s' % str(np.mean(np.array(acc))))
    print('averate acc:%s' % str(np.mean(np.array(acc))))

def decode_video(video_path, model_path, data, threshold=0.8):
    model = RivaGAN.load(model_path)
    acc = []
    frames = []
    clips = []
    threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    frames_list = [[] for _ in range(len(threshold_list))]
    i = 0
    window_size = 5
    for recovered_data in model.decode(video_path):
        acc.append(get_acc(data, recovered_data))

    for index, threshold in enumerate(threshold_list):
        frames = []
        for i, ac in enumerate(acc):
            if ac > threshold:
                frames.append(i)
        frames_list[index].extend(frames)
        # print('threshold:%s, frames:%s'%(str(threshold), str(frames)))
        clips = []
        if len(frames) < 1:
            pass
        if len(frames) == 1:
            clips.append([frames[0], frames[0]])
        if len(frames) > 1:
            start = frames[0]
            end = frames[0]
            for i in range(1,len(frames)):
                if frames[i] - frames[i-1] > window_size:
                    end = frames[i-1]
                    clips.append([start, end])
                    start = frames[i]
            clips.append([start, frames[-1]])
        print('threshold:%s, clips:%s, rate:%s'%(str(threshold),
                                                 str(clips),
                                                 str(len(frames)/len(acc))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="help")
    parser.add_argument('-i', '--input', help='the path of input file')
    parser.add_argument('-m', '--model', default='./model.pt', help='the path of model')
    parser.add_argument('-en', '--encode', action='store_true')
    parser.add_argument('-v', '--val', action='store_true', help='the encoded file you want to val')
    parser.add_argument('-de', '--decode', action='store_true', help='the file you want to decode')
    parser.add_argument('-o', '--output', help='output file path')
    parser.add_argument('-p', '--data', help='the data you want to encode')
    parser.add_argument('-s', '--datadim', help='the data size of password')

    args = parser.parse_args()

    model = args.model
    file_path = args.input
    encode = args.encode
    decode = args.decode
    val = args.val
    output_path = args.output
    data = args.data
    datadim = args.datadim

    if not os.path.exists(model):
        print('pleace specify valid model!')
        sys.exit(-1)
    if file_path is None:
        print('please choose to encode or decode')
        sys.exit(-1)
    if encode is False and decode is False and val is False:
        print('please choose your option(encode, decode or val)')
    if data is None:
        print('please input data(0~4294967296)!')
        sys.exit(-1)
    if datadim is None:
        print('please input data dim!(32 or 64)')
        sys.exit(-1)
    datamax = 18446744073709551616 if int(datadim) == 64 else 4294967296
    if data is not None and (int(data) < 0 or int(data) >= datamax):
        print('please input valid data(0~%s)'%str(datamax))
        sys.exit(-1)
    data = list(bin(int(data))[2:].zfill(int(datadim)))
    data = [int(d) for d in data]

    if encode:
        if output_path is None:
            print('please input output path!')
            sys.exit(-1)
        num, den = get_frame_rate(file_path)
        if num is None or den is None:
            print('your file can\'t get frame rate!')
            sys.exit(-1)
        fps = round(num / den, 2)
        encode_video(file_path, output_path, model, data, fps)
        if val:
            val_decode_video(output_path, model, data)

    if decode:
        decode_video(file_path, model, data)
        # for index, clip in enumerate(clips):
        #     print('clip %d:[%d, %d]'%(index, clip[0], clip[1]))

    if val and not encode:
        val_decode_video(file_path, model, data)


