You need menagerie files in workflow directory.

How to install

git clone https://github.com/google-deepmind/mujoco_menagerie.git

Other packages:
```bash
python -m pip install numpy pandas opencv-python mujoco glfw scikit-image Pillow torch torchvision
```


Command line example:

Image processing
```bash
python photo_to_bw_human_cartoon_fixed_v5.py LM.png     -o LM_bw_cartoon_fixed.png     --render-mode portrait     --portrait-style ink     --detail medium     --line-width 2     --minimum-component 18     --save-debug
```
or
```bash
python photo_to_bw_human_cartoon_fixed_v5.py LM.png     -o LM_bw_cartoon_simple.png     --render-mode portrait     --portrait-style ink     --detail medium     --line-width 1     --minimum-component 18     --save-debug
```

Robot simulation
```bash
(rcontrol) jy23choi@xxxxxxxxx:~/Desktop/panda_graph_mujoco_project/workflow$ python3 run_vertical_pen_track_recovery_20260806.py LM_bw_cartoon_simple.png --project-root /home/jy23choi/Desktop/panda_graph_mujoco_project/workflow --output-dir output/lm_vertical_track_recovery_simple --model model/drawing_scene.xml --scene-config model/drawing_scene_config.json --black-threshold 160 --minimum-component-size 1 --black-mask-close-iterations 0 --line-overlap 0.50 --global-grid-phase 0.50 --coverage-repair-iterations 12 --minimum-centerline-length 0.00005 --paper-center-x 0.50 --paper-center-y 0.00 --paper-width 0.32 --paper-height 0.20 --paper-margin 0.01 --spacing 0.001 --pen-spring-stiffness 70 --pen-spring-damping 1.0 --pen-spring-travel 0.015 --pen-body-radius 0.0025 --pen-tip-radius 0.0003 --pen-paper-penetration 0.00020 --guide-press-depth 0.0020 --lower-contact-gap-tolerance 0.0005 --contact-settle-time 0.30 --target-contact-force 0.05 --overforce-limit 7.0 --pose-completion-tolerance 0.001 --hard-pose-failure-tolerance 0.006 --pose-retries 3 --entry-xy-tolerance 0.00001 --entry-along-track-tolerance 0.00010 --endpoint-xy-tolerance 0.00001 --endpoint-along-track-tolerance 0.00010 --cross-track-slowdown-error 0.00018 --cross-track-stop-error 0.00055 --tracking-slowdown-error 0.0010 --tracking-stop-error 0.0040 --tracking-stall-timeout 8.0 --continuous-stroke-timeout-factor 4.0 --cross-track-position-gain 24 --along-track-position-gain 12 --normal-position-gain 10 --cartesian-damping 0.015 --seat-correction-attempts 4 --maximum-seat-correction-depth 0.0025 --seat-correction-margin 0.00010 --draw-start-settle-time 0.30 --draw-end-settle-time 0.20 --xy-stable-time 0.03 --draw-speed 0.0005 --transfer-speed 0.003 --vertical-speed 0.01 --log-stride 10 --progress-width 18 --no-viewer --rebuild-spring-model
```
Switch --no-viewer- to --viewer in the command line if you want to see the whole MuJoCo 3D robot simulation


Console window should be like this:

[black_pixels_to_vertical_graph]
[vertical_graph_to_paths]
[mujoco_spring_pen_simulation]
MODEL BUILD: panda_vertical_track_recovery_20260806
Fixed Panda base and original desk/paper: preserved
Gripper guide body: hand
Preserved and spring-mounted original rigid pen parts: ['drawing_pen_body', 'drawing_pen_tip', 'pen_tip']
Pen dimensions: body diameter=0.0050 m, tip diameter=0.0006 m
Passive spring: joint=spring_pen_joint, stiffness=70 N/m, damping=1 N s/m, range=[-0.015000, 0.002000] m
Paper geom=paper_geom, top z=0.204000 m
Generated model: /home/jy23choi/Desktop/panda_graph_mujoco_project/workflow/output/lm_vertical_track_recovery_simple/runtime_model/drawing_scene_vertical_track_recovery_20260806.xml
SIMULATOR BUILD: panda_vertical_track_recovery_20260806
Fixed Panda base, original desk, and original paper: enabled
Control split: joints 1-7 move the rigid guide block through exact vertical-fill XY; the pen slides only through the passive spring joint inside the guide block.
Trajectory source: /home/jy23choi/Desktop/panda_graph_mujoco_project/workflow/output/lm_vertical_track_recovery_simple/drawing_strokes.csv | vertical-fill path points=13116 | path_geometry=dense_vertical_line_pen_width_track_recovery
Paper=paper_geom, top z=0.204000 m; physical contact tip-center z=0.204100 m; direct-press command z=0.202100 m; scene-config draw z=0.204800 m
SCENE CONFIG NOTE: draw_target_tip_center_z_m describes the source pen; the compiled runtime tip geometry is authoritative for this run.
Pen body diameter=0.0050 m; physical tip diameter=0.0006 m
Spring stiffness=70 N/m, target force=0.05 N, target compression=0.000714 m, target q=-0.000714 m, dz/dq=-1.000000
Vertical-line tracking control: approach/lift tolerance=0.001000 m; entry cross/along=0.000010/0.000100 m; cross slowdown/stop=0.000180/0.000550 m; along slowdown/stop=0.001000/0.004000 m
Continuous stroke streaming: 13116 prepared waypoints are interpolation knots, not stop targets; draw length=10.277 m, nominal draw time=5:42:34, nominal XY transfer time=4:19:04
Opening interactive MuJoCo viewer...
MuJoCo viewer opened and synchronized (backend=auto).
Drawing [##################] 16874/16874 100.0% elapsed 47:48 | complete
Simulation outputs written: simulated_drawing.png, simulated_trajectory.png, simulation_summary.json; contact_ink_ok=True; notices=7524
Motion diagnostics: /home/jy23choi/Desktop/panda_graph_mujoco_project/workflow/output/lm_vertical_track_recovery_simple/skipped_strokes_record.json
Manifest: /home/jy23choi/Desktop/panda_graph_mujoco_project/workflow/output/lm_vertical_track_recovery_simple/workflow_manifest.json
