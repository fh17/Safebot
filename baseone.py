from estimater import *
from frameprocessing import reader
from camerareader import *
from YOLOv11seg import *
import argparse
import subprocess



if __name__=='__main__':
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
 
  K = np.loadtxt(f'{args.test_scene_dir}/cam_K.txt').reshape(3,3)
  W, H =  int(K[0, 2] * 2), int(K[1, 2] * 2)  # around (640, 480) #
  zfar = np.inf  # Update if necessary

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = reader(video_dir=args.test_scene_dir, H=H, W=W,shorter_side=None, zfar=np.inf)


  camera = CameraReader("Camera 1")
  mask_generator = MaskGenerator()

  if not camera.initialized:
     print("Camera initialization failed. Exiting.")
     exit(1)  
  i = 0
  # we get depth and color frames already processed from frameprocessing
  depth, color = reader.get_frame(camera)

  if depth is not None and color is not None:
    if i==0:
#..................................................................................................................................................................

      cv2.imwrite(f'{args.test_scene_dir}/picformask.png', color)
      cv2.imwrite(f'{args.test_scene_dir}/depthpic.png', depth)
#..................................................................................................................................................................

      mask = mask_generator.process_image(f'{args.test_scene_dir}/picformask.png')
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
      cv2.imwrite(f'{args.test_scene_dir}/picmask.png', mask)
#..................................................................................................................................................................

      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      #save first estimation homogeneous transformation matrix
      np.savetxt(f'{args.test_scene_dir}/first_est_pose_matrix.txt', pose, fmt='%f')
#..................................................................................................................................................................


      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        
        i= i+1

    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

   # i = i+1

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)



