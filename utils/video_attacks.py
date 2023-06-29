import os
import glob
import cv2
import av


def video_compression_attack(encode_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, interval):
    logger.info("Start Compressing :")

    length = len(glob.glob(os.path.join(encode_folder, 'frame-*.png')))
    # images = [cv2.imread(os.path.join(encode_folder, 'frame-%04d.png' % i)) for i in range(length)]
    images = [os.path.join(encode_folder, 'frame-%04d.png' % i) for i in range(length)]
    image_tmp = cv2.imread(images[0])

    output = av.open(compress_folder + '.mp4', 'w')
    stream = output.add_stream(compress_type, fps)

    stream.height = int(float(resize_ratio) * image_tmp.shape[0])//2*2
    stream.width = int(float(resize_ratio) * image_tmp.shape[1])//2*2
    stream.options = {'crf': crf, 
                      'keyint_min': str(interval), 
                      'g': str(interval),
                      'x264-params'      : 'scenecut=0'}
    logger.info("{0}: fps-{1} | crf-{2} | HxW-{3}x{4}", compress_type, fps, crf, stream.height, stream.width)

    for i, img in enumerate(images):
        img = cv2.imread(img)
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

    output.mux(stream.encode())
    output.close()
    container = av.open(compress_folder + '.mp4')

    if not os.path.exists(compress_folder):
        os.mkdir(compress_folder)
    frame_id = 0
    for frame in container.decode(video=0):
        img = frame.to_image()
        img.save(os.path.join(compress_folder, 'frame-%04d.png' % frame_id))
        frame_id += 1
    container.close()


def image_compress_attack(encode_folder, compress_folder, quality_factor, logger):
    logger.info("Start Compressing :")
    if not os.path.exists(compress_folder):
        os.mkdir(compress_folder)
    length = len(glob.glob(os.path.join(encode_folder, 'frame-*.png')))
    images = [os.path.join(encode_folder, 'frame-%04d.png' % i) for i in range(length)]
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(compress_folder, 'frame-%04d.jpg' % i), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])


def video_compression_attack2(encode_folder, compress_folder, crf, resize_ratio, fps, compress_type, logger, shift=10):
    logger.info("Start Compressing :")

    length = len(glob.glob(os.path.join(encode_folder, 'frame-*.png')))
    # images = [cv2.imread(os.path.join(encode_folder, 'frame-%04d.png' % i)) for i in range(length)]
    images = [os.path.join(encode_folder, 'frame-%04d.png' % ((i+shift)%length)) for i in range(length)]
    image_tmp = cv2.imread(images[0])

    output = av.open(compress_folder + '.mp4', 'w')
    stream = output.add_stream(compress_type, fps)

    stream.height = int(float(resize_ratio) * image_tmp.shape[0])//2*2
    stream.width = int(float(resize_ratio) * image_tmp.shape[1])//2*2
    stream.options = {'crf': crf}
    logger.info("{0}: fps-{1} | crf-{2} | HxW-{3}x{4}", compress_type, fps, crf, stream.height, stream.width)

    for i, img in enumerate(images):
        img = cv2.imread(img)
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

    output.mux(stream.encode())
    output.close()
    container = av.open(compress_folder + '.mp4')

    if not os.path.exists(compress_folder):
        os.mkdir(compress_folder)
    frame_id = 0
    for frame in container.decode(video=0):
        img = frame.to_image()
        img.save(os.path.join(compress_folder, 'frame-%04d.png' % frame_id))
        frame_id += 1
    container.close()