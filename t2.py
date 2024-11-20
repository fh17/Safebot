import time
from estimater import *
from frameprocessing import reader
from YOLOv11seg import *
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/data/live_stream/box/mesh/mesh.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/data/live_stream/box')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    K = np.loadtxt(f'{args.test_scene_dir}/cam_K.txt').reshape(3, 3)
    W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)  # around (640, 480) #
    zfar = np.inf  # Update if necessary

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                         refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = reader(video_dir=args.test_scene_dir, H=H, W=W, shorter_side=None, zfar=np.inf)

    mask_generator = MaskGenerator()

    # Frame capturing and processing
    i = 0
    depth, color = reader.get_frame(input)

    if depth is not None and color is not None:
        cv2.imwrite(f'{args.test_scene_dir}/picformask.png', color)
        cv2.imwrite(f'{args.test_scene_dir}/depthpic.png', depth)

        # Process the saved image to get the mask
        mask = mask_generator.process_image(f'{args.test_scene_dir}/picformask.png')

        cv2.imwrite(f'{args.test_scene_dir}/picmask.png', mask)

        # Continue with pose estimation

        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
