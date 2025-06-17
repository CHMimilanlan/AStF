import cv2
import imageio
from os.path import join as pjoin
import os
from pathlib import Path

def mp4_to_gif(input_path, output_path, fps=10, scale=1.0):
    """
    将MP4视频转换为GIF文件

    参数:
        input_path: 输入的MP4文件路径
        output_path: 输出的GIF文件路径
        fps: 输出GIF的帧率(默认10)
        scale: 缩放比例(默认1.0不缩放)
    """
    # 读取视频文件
    cap = cv2.VideoCapture(input_path)

    # 获取视频的原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算跳帧间隔以匹配目标fps
    frame_interval = max(1, int(round(original_fps / fps)))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 按间隔采集帧
        if frame_count % frame_interval == 0:
            # 转换颜色空间从BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 缩放图像
            if scale != 1.0:
                width = int(rgb_frame.shape[1] * scale)
                height = int(rgb_frame.shape[0] * scale)
                rgb_frame = cv2.resize(rgb_frame, (width, height))

            frames.append(rgb_frame)

        frame_count += 1

    cap.release()

    # 保存为GIF
    imageio.mimsave(output_path, frames, fps=fps)

    print(f"GIF保存成功: {output_path}")


# 使用示例
if __name__ == "__main__":
    basedir = "render_video"
    outpath = "assets"
    os.makedirs(outpath, exist_ok=True)
    # for i in range(4,5):
    #     numdir = pjoin(basedir, str(i))
    #     os.makedirs(pjoin(outpath, str(i)), exist_ok=True)
    #     for mp4 in os.listdir(numdir):
    #         mp4path = pjoin(numdir, mp4)
    #         print(f"process {mp4path}")
    #         name = Path(mp4path).stem
    #         output_gif = pjoin(pjoin(outpath, str(i)), f"{name}.gif")  # 输出的GIF文件名
    #         mp4_to_gif(mp4path, output_gif, fps=30, scale=0.5)

    for i in ["teaser"]:
        numdir = pjoin(basedir, str(i))
        os.makedirs(pjoin(outpath, str(i)), exist_ok=True)
        for mp4 in os.listdir(numdir):
            mp4path = pjoin(numdir, mp4)
            print(f"process {mp4path}")
            name = Path(mp4path).stem
            output_gif = pjoin(pjoin(outpath, str(i)), f"{name}.gif")  # 输出的GIF文件名
            mp4_to_gif(mp4path, output_gif, fps=30, scale=0.5)



    # basedir = "11"
    # outpath = "assets"
    # for mp4 in os.listdir(basedir):
    #     mp4path = pjoin(basedir, mp4)
    #     print(f"process {mp4path}")
    #     name = Path(mp4path).stem
    #     output_gif = pjoin(pjoin(outpath, f"{name}.gif"))  # 输出的GIF文件名
    #     mp4_to_gif(mp4path, output_gif, fps=25, scale=0.5)

    # mp4path = r"Style_depressed_01_000_Content_childlike_16_000.mp4"
    # output_gif = "Style_depressed_01_000_Content_childlike_16_000.gif"
    # mp4_to_gif(mp4path, output_gif, fps=25, scale=0.5)
