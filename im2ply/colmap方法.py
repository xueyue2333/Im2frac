import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=r'D:\A-Fkf\MVSNet_pytorch-master\new_data\scan7', help="the path contains the folder of images")
    parser.add_argument("--specify_intrinsic", action="store_true", help="whether specify the intrinsic for the scene")
    parser.add_argument("--use_auto", action="store_true", help="whether use colmap automatic_reconstructor")

    parser.add_argument("--max_num_features", type=int, default=8192)
    parser.add_argument("--max_error", type=int, default=1.00)

    args = parser.parse_args()
    base_dir = args.base_dir

    if args.use_auto:
        cmd = "colmap automatic_reconstructor --workspace_path {} --image_path {}/images".format(base_dir, base_dir)
        os.system(cmd)
    else:
        print("\n---------------特征提取------------\n")
        # if args.specify_intrinsic:
        #     cmd1 = "colmap feature_extractor \
        #             --ImageReader.camera_model PINHOLE\
        #             --database_path {}/database.db \
        #             --image_path {}/images".format(base_dir, base_dir)
        # else:
        cmd1 = "colmap feature_extractor \
                --ImageReader.camera_model PINHOLE\
                --ImageReader.single_camera 1 \
                --database_path {}/database.db \
                --SiftExtraction.max_num_features {}\
                --image_path {}/images".format(base_dir, args.max_num_features, base_dir)

        os.system(cmd1)

        print("\n---------------特征匹配------------\n")
        # cmd2 = "colmap sequential_matcher --database_path {}/database.db" \
        #        " --SiftMatching.max_error {}".format(base_dir, args.max_error)
        cmd2 = "colmap exhaustive_matcher --database_path {}/database.db" \
               " --SiftMatching.max_error {}".format(base_dir, args.max_error)
        os.system(cmd2)

        print("\n---------------稀疏建图与ba ------------\n")
        os.makedirs("{}/sparse".format(args.base_dir), exist_ok=True)
        cmd3 = "colmap mapper \
                --database_path {}/database.db \
                --image_path {}/images \
                --output_path {}/sparse".format(base_dir, base_dir, base_dir)
        os.system(cmd3)

        print("\n---------------畸变校正 ------------\n")
        os.makedirs("{}/dense".format(base_dir), exist_ok=True)
        cmd4 = "colmap image_undistorter \
                --image_path {}/images \
                --input_path {}/sparse/0 \
                --output_path {}/dense \
                --output_type COLMAP".format(base_dir, base_dir, base_dir)
        os.system(cmd4)

        cmd7 = r"colmap model_converter --input_path {}\dense\sparse --output_path {}\dense\sparse --output_type TXT".format(base_dir, base_dir)
        os.system(cmd7)
