(Note: part 1 verrified but need some extra steps)

--

mujoco, a robotic simulator, got acquired and then an open source version released. still, I feel like this field is not accessible to beginners - with recent LLM popularity, the topics of VLA, humanoid robotics, and embodied intelligence (and world model) got popular. today, my goal is more modest - I remember working through chapter 1 of the official mujoco tutorial series ( a collection of jupyter notebook) and it feels very tedious. can you give me a guide/code/cheatsheet etc that skip the theory and just go straight to a working setup? prefer to couple with VLA model inference loop (eg the lerobot library etc) for a sim rollout that also record video. if the VLA part is out of your knowledge, you may assume it's a huggingface transformer model instead (eg pi 0.5 uses a VLM backbone for conditioning info + a transformer diffusion model for the action space I think). thanks!

---
<think>
The user wants a practical guide: MuJoCo setup, skipping theory, plus a VLA inference loop (lerobot-style) with video recording. They want a cheatsheet/code. Today is 2026-08. Let me write a comprehensive but practical guide.

Key points:

    MuJoCo is now open source (DeepMind acquired, Apache 2.0 since 2022), pip installable.
    Key packages: mujoco, mujoco-python-viewer or dm_control, gymnasium (mujoco envs), lerobot from Hugging Face.
    LeRobot has LeRobotDataset, policies like Pi0, ACT, Diffusion Policy, SmolVLA; lerobot pip install; policies can be loaded from HF hub with PreTrainedPolicy.from_pretrained.
    Sim rollout: use gymnasium env (e.g., PushT, Aloha sim env in lerobot: lerobot/common/envs — actually lerobot has gym_aloha, gym_pusht packages). Also lerobot-record CLI.
    Video recording: gymnasium RecordVideo wrapper, or env.render(), or lerobot's record.

Provide:

    Minimal install commands.
    Minimal MuJoCo hello world (no theory): load model, step, render offscreen to video.
    Gymnasium + MuJoCo envs with RecordVideo.
    VLA inference loop: generic template — load policy from HF hub via lerobot, get observation dict (image, state), predict action chunk, step env, record video. Include both lerobot-specific code and a generic "assume it's a HF transformer" fallback.
    Cheatsheet table of key APIs.

Let me write concrete code.

Install:

pip install mujoco "gymnasium[mujoco]" imageio
# or dm_control
pip install lerobot  # from HF
pip install gym-pusht gym-aloha  # lerobot sim envs

MuJoCo hello world:

import mujoco
model = mujoco.MjModel.from_xml_path("...") or from_xml_string
data = mujoco.MjData(model)
viewer / renderer
renderer = mujoco.Renderer(model, height, width)
while ...:
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frame = renderer.render()
imageio.mimsave(...)

Gymnasium with video:

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda i: True)
obs, info = env.reset(seed=0)
for step in range(1000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc: break
env.close()

VLA loop with lerobot: Pi0/SmolVLA. lerobot API (as of ~2025):

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

Or generic:

from lerobot.common.policies.pretrained import PreTrainedPolicy

Actually lerobot has lerobot.common.policies module; with newer versions lerobot.policies.*. Policy call: policy.select_action(batch) where batch is dict with keys like observation.images.cam, observation.state, task (language string). For diffusion/Pi0, need to call policy.reset() at episode start, and predict action chunk then execute.

Generic rollout:

policy.reset()
for step:
    batch = {
        "observation.images.cam": torch.from_numpy(obs["pixels"]["agent"]).cuda(),
        "observation.state": torch.from_numpy(obs["agent_pos"]).cuda(),
        "task": ["push the t to the target"],
    }
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = action.squeeze(0).cpu().numpy()
    obs, reward, ... = env.step(action)

gym_pusht: gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos"). obs keys: "pixels" dict with "agent" and "top", "agent_pos".

Fallback generic HF transformer template: a wrapper class showing where to plug in a custom model.

Also mention lerobot-record CLI for recording datasets:

lerobot-record --robot-type pusht --policy.path lerobot/smolvla_base ...

Actually the CLI evolved; in 2025 it's python -m lerobot.scripts.record or lerobot-record after pip install lerobot[all]. Keep it approximate with a caveat.

Also dm_control alternative briefly.

Cheatsheet: key mujoco APIs table (MjModel, MjData, mj_step, mj_forward, Renderer, mj_name2id, actuator ctrl), gymnasium wrappers.

Note environment mismatch: policy trained on Aloha/PushT needs matching env & action space normalization.

Keep it well-organized. Length: fairly long but that's what's asked. Use code blocks heavily.
</think>

# MuJoCo + VLA Rollout: The "Skip-the-Theory" Guide

You're right that the official tutorial notebooks front-load a lot of physics theory. Here's the pragmatic path: install → render anything to video → plug in a policy.

---

## 0. What to install (2026 state of the world)

```bash
# Core stack — MuJoCo is Apache 2.0 since the DeepMind acquisition, pip-installable
pip install mujoco gymnasium[mujoco] imageio[ffmpeg]

# LeRobot (Hugging Face) — policies, datasets, sim env wrappers
pip install lerobot

# Sim envs LeRobot policies were actually trained on
pip install gym-pusht gym-aloha
```

Notes:
- No more `mujoco-py`, no license key, no `apt install libgl1` dance (mostly). If `mujoco` import fails on Linux, `apt install libosmesa6-dev libgl1-mesa-glx` usually fixes it.
- Verify:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

---

## 1. MuJoCo in 15 lines (the mental model is 5 words)

The entire MuJoCo API surface you need day-to-day:

| Object | What it is |
|---|---|
| `MjModel` | the *static* description (parsed XML: bodies, joints, actuators) |
| `MjData` | the *dynamic* state (qpos, qvel, forces) — what changes each step |
| `mj_step(model, data)` | advance physics by one timestep |
| `Renderer` | offscreen RGB rendering → numpy frames (this is how you get video) |

```python
import mujoco, numpy as np, imageio

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 .1"/>
    <body pos="0 0 .5">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

frames = []
for _ in range(200):
    mujoco.mj_step(model, data)          # physics
    renderer.update_scene(data)          # sync camera
    frames.append(renderer.render())     # HxWx3 numpy

imageio.mimsave("drop.mp4", frames, fps=int(round(1/model.opt.timestep)))
```

That's it. **Control = write to `data.ctrl`, observe = read `data.qpos`.** Everything else in the tutorial notebooks is detail.

Quick reference for when you need it:
```python
data.ctrl[:] = action            # set actuator commands
qpos = data.qpos.copy()          # joint positions
mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "name")  # look things up by name
mujoco.mj_forward(model, data)   # recompute kinematics WITHOUT stepping (needed before rendering if you set state manually)
```

---

## 2. Gymnasium route (recommended — this is what VLAs expect)

Don't hand-roll the loop against raw MuJoCo unless you're building a custom robot. Use the Gymnasium API, which every policy library (LeRobot, SB3, etc.) speaks natively. Video recording is one wrapper:

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True,
                  name_prefix="random-policy")

obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()   # replace with policy output later
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()  # REQUIRED — video is flushed on close
```

Gotchas:
- `render_mode="rgb_array"` must be passed at `gym.make`, not later.
- `env.close()` writes the mp4. Forgetting this is the #1 "where's my video" bug.
- Frame rate mismatch: if video looks slow/fast, `RecordVideo(..., fps=env.metadata["render_fps"])`.

---

## 3. The VLA inference loop (LeRobot-style)

LeRobot policies (`SmolVLA`, `Pi0`, `ACT`, `DiffusionPolicy`...) all follow the same contract, which is what you guessed: a VLM/transformer backbone conditioned on task language + images, emitting an **action chunk** (e.g., 32-50 future actions) executed one at a time. The loop:

1. `policy.reset()` at episode start (clears the action-chunk queue)
2. Build an observation **batch dict** (note the `policy` prefix keys, LeRobot convention)
3. `policy.select_action(batch)` → execute ONE action in the env
4. Repeat; library internally replays the cached chunk and re-predicts when exhausted

```python
import gymnasium as gym, torch, numpy as np
from gymnasium.wrappers import RecordVideo
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# --- load pretrained policy from the HF Hub ---
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.to(device)

# --- env: PushT is the "hello world" of VLAs ---
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

obs, info = env.reset(seed=0)
policy.reset()  # clears internal action queue — important!

episode_return = 0.0
for step in range(300):
    # --- observation batch, matching the policy's training keys ---
    batch = {
        "observation.images.top":   torch.from_numpy(obs["pixels"]["top"]).to(device),   # H,W,3 uint8
        "observation.state":        torch.from_numpy(obs["agent_pos"]).float().to(device),
        "task": ["push the T-shaped block onto the target circle"],  # language conditioning
    }

    with torch.inference_mode():
        action = policy.select_action(batch)          # (action_dim,) tensor
    action = action.squeeze(0).cpu().numpy()

    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:
        break

env.close()
print(f"return: {episode_return:.2f}")
```

**The three classic failure modes** (in order of how often they bite people):

1. **Key mismatch.** Each policy expects specific `observation.images.<cam_name>` / `observation.state` keys. Inspect with `policy.config.input_features` (or just `print(policy.config)`). A policy trained on Aloha's `wrist` + `top` cameras won't accept `agent`.
2. **Observation normalization.** Some checkpoints expect images in `[-1, 1]` or states standardized. LeRobot's `from_pretrained` usually stores normalization stats in the checkpoint and applies them — but if you're wiring a raw HF transformer (see §4), you must handle this yourself.
3. **Action space mismatch.** Action dims and ranges must match the env (`env.action_space`). Pi0-style models output normalized actions in `[-1, 1]`; check whether the env expects that or raw units.

Bonus: LeRobot also has a CLI that does record-episodes-with-a-policy for you, worth knowing:

```bash
# record eval rollouts of a pretrained policy (exact flags vary by version — lerobot --help)
lerobot-record \
  --robot.type=pusht \
  --dataset.repo_id=local/eval_rollouts \
  --policy.path=lerobot/smolvla_base \
  --num-episodes=10 --push-to-hub=0
```

---

## 4. Fallback: plugging in an arbitrary HF transformer (your "assume it's a HF model" case)

If you're not using LeRobot policies but a custom Pi0.5-ish model (VLM backbone + flow/diffusion action head), wrap it behind the same interface so your eval loop stays identical:

```python
import torch
from transformers import AutoModel, AutoProcessor

class VLAPolicy:
    """Adapter: any HF transformer emitting action chunks → gym-compatible `select_action`."""
    def __init__(self, model_id: str, device="cuda", chunk_size: int = 32):
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.device, self.chunk_size = device, chunk_size
        self._queue = None

    def reset(self):
        self._queue = None

    @torch.inference_mode()
    def select_action(self, obs: dict, task: str) -> "np.ndarray":
        # replay cached chunk until exhausted ( amortizes the expensive VLM forward pass )
        if self._queue is None or len(self._queue) == 0:
            inputs = self.processor(
                images=[obs["image"]],            # your camera frame
                state=obs["state"],               # proprioception
                text=task, return_tensors="pt"
            ).to(self.device)
            # typical Pi0-style head: (B, chunk_size, action_dim), usually in [-1, 1]
            self._queue = self.model(**inputs).actions.squeeze(0).cpu().numpy()
        action = self._queue[0]
        self._queue = self._queue[1:]
        return action

# Usage — identical loop shape as §3:
policy = VLAPolicy("your-org/your-pi0-model")
obs, _ = env.reset(seed=0); policy.reset()
for _ in range(300):
    action = policy.select_action({"image": frame, "state": obs["agent_pos"]},
                                  "pick up the red block")
    obs, r, term, trunc, _ = env.step(action)
    if term or trunc: break
```

The key design point: **predict a chunk, execute one action per env step.** This is universal across Pi0/π0.5, OpenVLA-variants, SmolVLA, etc. — running the VLM every single step is both too slow and what the models weren't trained for.

---

## 5. TL;DR cheatsheet

```
INSTALL    pip install mujoco gymnasium[mujoco] imageio[ffmpeg] lerobot gym-pusht gym-aloha

RAW MJ     model = MjModel.from_xml_path/string(...)
           data  = MjData(model)
           loop: data.ctrl[:] = a; mj_step(model, data); Renderer(model,...).render()

GYM        env = gym.make("X-v5", render_mode="rgb_array")
           env = RecordVideo(env, "videos", episode_trigger=lambda e: True)
           ... env.close()   # flushes the mp4!

VLA        policy = XPolicy.from_pretrained("lerobot/...")
           per episode: policy.reset()
           per step:    action = policy.select_action({
                           "observation.images.top": img_tensor,
                           "observation.state":      state_tensor,
                           "task":                   [language_str] })
           action → env.step(action)

DEBUG      print(policy.config)          # expected obs keys / action dim
           env.action_space              # sanity check dims & bounds
           frames too slow → RecordVideo(..., fps=env.metadata["render_fps"])
```

If you want a next step after this works: swap PushT for the Aloha sim env (`gym.make("gym_aloha/AlohaInsertion-v0")`) to test bimanual manipulation, or fine-tune SmolVLA on your own recorded episodes with the LeRobot training scripts. Happy to go deeper on any of these once you have the basic rollout running.



----

very good! can you design custom mujoco env (maybe wrap as gymnasium task) for some robotics scenarios? one easy and one moderate difficulty. maybe something like pickup the dish on desk and put on dishwasher, or walking to take a vacuum cleaner, that kind of stuff. (if full humanoid is too hard, you may model a vastly simplified version that is "inspired" by the mentioned scenario and keep the spirit but say the robot is just a two joints arm with movable base etc)

----

# Custom MuJoCo Envs (Gymnasium-Wrapped): Two Worked Examples

Great next step. Below are two complete, self-contained environments, both following the same skeleton so you can see the pattern. I'll take your suggestion: **the "vastly simplified but honest" approach** — full humanoids and dish racks are week-long projects; a 2-3 DoF arm with a graspable object teaches you 95% of the concepts.

- **Easy: `DishToSink`** — a 2-joint planar arm must push/grasp a "dish" (a puck) from a table into a sink basin. Reach, contact, sparse-ish reward.
- **Moderate: `FetchVacuum`** — a mobile base (prismatic x, y) carrying a 3-joint arm must navigate to a vacuum cleaner lying on the floor, grab it, and carry it to a target zone. Multi-stage task, success/failure logic, curriculum-friendly rewards.

---

## 0. The universal recipe for a custom MuJoCo Gymnasium env

Every env you'll ever write is this same skeleton:

```
1. Write MJCF XML (the scene: bodies, joints, actuators, sensors, sites)
2. __init__:   parse XML → MjModel, MjData, Renderer; define spaces
3. reset:      randomize object/goal poses, mj_resetData + mj_forward
4. step:       data.ctrl[:] = action → mj_step → compute obs/reward/done
5. render:     renderer.update_scene(data); return renderer.render()
```

Key design decisions you make per-env:

| Decision | Easy default | Why |
|---|---|---|
| Actuation | `<motor>` (torque) or `<position>` servos | Position servos ≈ what real robot arms do, easier to control |
| Observation | qpos of arm + xpos of objects (+ optional pixels) | VLAs need pixels + state; classic RL just needs state |
| Reward | staged: approach → grasp/push → goal | sparse-only rewards are miserable to learn from |
| Contacts | use `<contact>` excludes, name your geoms | unnameed geoms = debugging hell |
| Termination | success condition + step budget (truncation) | `terminated` vs `truncated` matters for RL bootstrap |

One tooling tip before we start — install this helper for interactive debugging, it will save you hours:

```python
# viewer.py — launch any MjModel interactively (mouse orbit, joint sliders via double-click)
import mujoco, mujoco.viewer
model = mujoco.MjModel.from_xml_path("your_scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)          # blocking, interactive
# or non-blocking inside your own loop:
# with mujoco.viewer.launch_passive(model, data) as v: ... v.sync()
```

---

## 1. EASY: `DishToSink-v0` — 2-joint arm, push dish into sink

**Scene**: planar arm (shoulder + elbow, position-controlled) on a table. A dish (flat cylinder) sits at a random spot. A "sink" is a recessed basin at the table edge. Win = dish inside the sink region and (nearly) at rest.

The robot never "grasps" here (we have no gripper) — it **pushes** the dish, like a shuffleboard. That's actually pedagogically nice: pushing is a well-studied manipulation primitive (cf. the classic PushT you just ran).

### 1.1 The MJCF XML

```python
# dish_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

DISH_XML = """
<mujoco model="dish_to_sink">
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <compiler angle="radian"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>

  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom friction="0.8 0.005 0.0001" condim="3"/>
    <position kp="40"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>

    <!-- table -->
    <body name="table" pos="0 0 0.4">
      <geom name="tabletop" type="box" size="0.6 0.4 0.02" rgba="0.75 0.6 0.45 1"/>
    </body>

    <!-- sink: a recessed basin modeled as walls at the table edge -->
    <body name="sink" pos="0.45 0 0.4">
      <geom name="sink_walls" type="box" size="0.10 0.15 0.03" rgba="0.55 0.55 0.6 1"
            pos="0 0 0.045"/>
      <!-- visually mark the target region with a flat red zone -->
      <geom name="sink_zone" type="box" size="0.09 0.14 0.002" rgba="0.9 0.1 0.1 0.3"
            pos="0 0 0.021" contype="0" conaffinity="0"/>
    </body>

    <!-- the dish: free-floating flat puck -->
    <body name="dish" pos="0 0 0.43">
      <freejoint name="dish_joint"/>
      <geom name="dish_geom" type="cylinder" size="0.05 0.008" mass="0.1"
            rgba="0.95 0.95 0.9 1"/>
    </body>

    <!-- the arm: 2 revolute joints in the x-z plane (planar) -->
    <body name="arm_base" pos="-0.3 0 0.42">
      <geom name="base_geom" type="box" size="0.05 0.05 0.05" rgba="0.2 0.2 0.2 1"/>
      <body name="upper_arm" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="upper_geom" type="capsule" fromto="0 0 0 0.3 0 0"
              size="0.02" mass="0.5" rgba="0.2 0.6 0.9 1"/>
        <body name="fore_arm" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="fore_geom" type="capsule" fromto="0 0 0 0.25 0"
                size="0.015" mass="0.3" rgba="0.3 0.7 0.95 1"/>
          <site name="end_effector" pos="0.25 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="shoulder_servo" joint="shoulder"/>
    <position name="elbow_servo" joint="elbow"/>
  </actuator>

  <sensor>
    <!-- object pose sensors: easy obs wiring, no manual indexing -->
    <framepos name="dish_pos" objtype="body" objname="dish"/>
    <framequat name="dish_quat" objtype="body" objname="dish"/>
    <framelinvel name="dish_vel" objtype="body" objname="dish"/>
  </sensor>
</mujoco>
"""
```

Design notes (the parts tutorials don't tell you):
- **Position actuators, not motors.** `<position kp="40"/>` means `data.ctrl = [target_shoulder, target_elbow]` in radians and MuJoCo's built-in PD does the rest. Your action space becomes "joint targets in [-1,1] mapped to [-π, π]" — much friendlier for both RL and VLAs.
- **Sensors for observations.** `<framepos>`/`<framequat>` on named bodies beats reading `data.qpos` indices by hand. `data.sensor("dish_pos").data` — self-documenting.
- **The sink is walls, not a hole.** Cutting a hole in the tabletop mesh is overkill; three low walls + a red zone geom (with `contype=0` so it's visual-only) reads as "sink" for both humans and pixel-observing policies.

### 1.2 The Gymnasium wrapper

```python
class DishToSinkEnv(gym.Env):
    """Push the dish into the sink with a 2-DoF planar arm.
    Action: [shoulder_target, elbow_target] in [-1,1] -> radians.
    Obs:    [shoulder, elbow, dish_x, dish_z, dish_quat(4), dish_vel(2)]
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    SINK_CENTER = np.array([0.45, 0.0, 0.43])
    SINK_TOL    = 0.07      # dish center within this of sink center
    MAX_STEPS   = 300

    def __init__(self, render_mode="rgb_array", camera_name=None, image_obs=False):
        self.model = mujoco.MjModel.from_xml_string(DISH_XML)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._camera_name = camera_name
        self.image_obs = image_obs

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        if image_obs:
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                "pixels": spaces.Dict({
                    "top": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                }),
            })
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

    # ---- helpers -------------------------------------------------
    def _get_obs(self):
        state = np.concatenate([
            self.data.qpos[:2],                          # shoulder, elbow
            self.data.sensor("dish_pos").data,           # xyz
            self.data.sensor("dish_quat").data,          # quat
            self.data.sensor("dish_vel").data[[0, 2]],   # vx, vz (planar)
        ]).astype(np.float32)
        if self.image_obs:
            return {"agent_pos": state, "pixels": {"top": self.render()}}
        return state

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self._camera_name)
        return self._renderer.render()

    def _dish_in_sink(self):
        pos = self.data.sensor("dish_pos").data
        vel = np.linalg.norm(self.data.sensor("dish_vel").data)
        return (np.linalg.norm(pos - self.SINK_CENTER) < self.SINK_TOL) and (vel < 0.05)

    # ---- core API ------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # random dish start position on the reachable table area
        x = self.np_random.uniform(-0.15, 0.25)
        self.data.qpos[self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "dish_joint")]:][:3] = [x, 0, 0.43]
        self.data.qpos[:2] = [0.5, -1.0]     # arm home pose

        mujoco.mj_forward(self.model, self.data)   # CRITICAL after manual qpos edits
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # map [-1,1] -> joint target radians
        self.data.ctrl[:] = np.clip(action, -1, 1) * np.pi
        mujoco.mj_step(self.model, self.data)
        self._steps += 1

        # ---- staged reward (dense enough to learn from) ----
        dish_pos = self.data.sensor("dish_pos").data
        dist = np.linalg.norm(dish_pos - self.SINK_CENTER)

        reward = -0.1 * dist                       # approach
        if dist < 0.15:
            reward += 0.5                          # close enough to push in
        if self._dish_in_sink():
            reward += 10.0                         # success bonus

        # dish fell off the table entirely -> penalty
        if dish_pos[2] < 0.30:
            reward -= 5.0

        terminated = self._dish_in_sink()
        truncated = self._steps >= self.MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, {"success": terminated}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


# ---- registration so `gym.make("DishToSink-v0")` works ----
from gymnasium.envs.registration import register
register(id="DishToSink-v0", entry_point=dish_env_module_path := __name__ + ":DishToSinkEnv")
```

### 1.3 Sanity-check it immediately (do this before any training!)

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

env = gym.make("DishToSink-v0", render_mode="rgb_array")
env = RecordVideo(env, "videos", episode_trigger=lambda e: True, name_prefix="dish")

obs, _ = env.reset(seed=0)

# scripted "policy": sweep the arm toward the sink — proves physics + contacts work
for t in range(300):
    phase = min(t / 150.0, 1.0)
    action = np.array([0.5 - 0.6 * phase, -1.0 + 0.3 * phase], dtype=np.float32)
    obs, r, term, trunc, info = env.step(action)

env.close()  # flush video; watch videos/dish-episode-0.mp4
```

If the video shows the arm sweeping and the dish sliding when hit — your env is correct. **Never train anything on an env you haven't watched a scripted episode of.** (That single rule would have saved me weeks.)

---

## 2. MODERATE: `FetchVacuum-v0` — mobile base + 3-joint arm, multi-stage task

**Scene (honest simplification of "walk over and get the vacuum")**: a base that translates in x-y (think of it as a very dedicated pair of legs), a 3-DoF arm on top (yaw + shoulder + elbow), and a gripper simplified to a **magnetic attachment**: when the end-effector site is close to the vacuum's handle and the "grip" action channel is on, the vacuum attaches. Carry it to the marked charging dock.

Three stages: **navigate → attach → deliver**. This introduces: multi-subtask rewards, stage tracking, an attach mechanic (stand-in for grasping), and a 5-dim action space.

### 2.1 MJCF XML

```python
VACUUM_XML = """
<mujoco model="fetch_vacuum">
  <option timestep="0.01"/>

  <compiler angle="radian"/>
  <default>
    <joint damping="0.5" armature="0.1"/>
    <position kp="60"/>
    <geom friction="1.0 0.005 0.0001"/>
  </default>

  <worldbody>
    <light pos="1 1 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1"/>

    <!-- the goal: charging dock, visual-only zone -->
    <body name="dock" pos="1.5 1.5 0.01">
      <geom name="dock_zone" type="cylinder" size="0.3 0.01" rgba="0.1 0.8 0.1 0.35"
            contype="0" conaffinity="0"/>
    </body>

    <!-- ============ the robot ============ -->
    <body name="robot" pos="-1 -1 0.05">
      <!-- mobile base: slides in x and y (our "walking") -->
      <joint name="base_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="base_y" type="slide" axis="0 1 0" range="-2 2"/>
      <geom name="chassis" type="cylinder" size="0.18 0.05" mass="5"
            rgba="0.25 0.25 0.3 1"/>

      <!-- turret: yaw -->
      <body name="turret" pos="0 0 0.06">
        <joint name="yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="turret_geom" type="cylinder" size="0.1 0.04" mass="1"/>

        <!-- shoulder -->
        <body name="upper_arm" pos="0 0 0.04">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="upper_geom" type="capsule" fromto="0 0 0 0 0 0.35"
                size="0.025" mass="1" rgba="0.2 0.6 0.9 1"/>

          <!-- elbow -->
          <body name="fore_arm" pos="0 0 0.35">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
            <geom name="fore_geom" type="capsule" fromto="0 0 0 0 0 0.3"
                  size="0.02" mass="0.5" rgba="0.3 0.7 0.95 1"/>
            <site name="gripper" pos="0 0 0.3" size="0.015"/>
          </body>
        </body>
      </body>
    </body>

    <!-- ============ the vacuum: free body with a handle site ============ -->
    <body name="vacuum" pos="1 0 0.06">
      <freejoint name="vacuum_joint"/>
      <geom name="vacuum_body" type="box" size="0.12 0.06 0.06" mass="2"
            rgba="0.9 0.75 0.1 1"/>
      <geom name="vacuum_handle" type="capsule" fromto="0 0 0.06 0 0 0.14"
            size="0.012" rgba="0.5 0.5 0.5 1"/>
      <site name="vacuum_grasp" pos="0 0 0.13" size="0.015" rgba="1 0 0 0.5"/>
    </body>
  </worldbody>

  <!-- ============ actuators ============ -->
  <actuator>
    <velocity name="drive_x" joint="base_x" kv="10"/>
    <velocity name="drive_y" joint="base_y" kv="10"/>
    <position  name="yaw_servo" joint="yaw"/>
    <position  name="shoulder_servo" joint="shoulder"/>
    <position  name="elbow_servo" joint="elbow"/>
  </actuator>

  <!-- ============ sensors ============ -->
  <sensor>
    <framepos name="robot_pos" objtype="body" objname="robot"/>
    <framepos name="vacuum_pos" objtype="body" objname="vacuum"/>
    <framepos name="gripper_pos" objtype="site" objname="gripper"/>
    <framepos name="dock_pos" objtype="body" objname="dock"/>
  </sensor>
</mujoco>
"""
```

New design decisions worth calling out:
- **`<velocity>` actuators for the base.** Now `data.ctrl[0:2]` = commanded base velocities in m/s. This is exactly how mobile bases / diff-drive robots work in sim, and it decouples "where am I" from "how do I steer" — much easier to learn than position control of slide joints spanning 4 meters.
- **The "gripper" is an equality-constraint trick**, shown below: we attach the vacuum to the end-effector by adding a `weld` constraint on demand. This is a standard simulation shortcut for grasp-and-carry (real grasp dynamics with contact-rich fingers are a whole other project).
- **Sites for grasp points.** `gripper` site on the arm, `vacuum_grasp` site on the handle — distance between two sites is the cleanest attach condition.

### 2.2 The Gymnasium wrapper

```python
class FetchVacuumEnv(gym.Env):
    """Navigate to the vacuum, attach it, carry it to the dock.

    Action (5-dim, all [-1,1]):
      [0:2]  base velocity command (x, y), scaled by 0.5 m/s
      [2:5]  joint targets (yaw, shoulder, elbow), scaled to radians
    Obs (14-dim):
      robot_xy, vacuum_xy, dock_xy, gripper_xyz, vacuum_z,
      joint angles (yaw, shoulder, elbow), attached flag
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    ATTACH_DIST = 0.05     # gripper site within this of grasp site
    DOCK_TOL    = 0.30
    MAX_STEPS   = 500

    def __init__(self, render_mode="rgb_array", image_obs=False):
        self.model = mujoco.MjModel.from_xml_string(VACUUM_XML)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self.image_obs = image_obs
        self.attached = False

        # cache ids we use every step (do this once, not per-step!)
        self._weld_id = None          # created dynamically on attach
        self._vacuum_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "vacuum_joint")
        self._vacuum_qposadr = self.model.jnt_qposadr[self._vacuum_jid]

        self.action_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        obs_dim = 14
        if image_obs:
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                "pixels": spaces.Dict({
                    "top": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
                }),
            })
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.sensor("robot_pos").data[:2],
            self.data.sensor("vacuum_pos").data[:2],
            self.data.sensor("dock_pos").data[:2],
            self.data.sensor("gripper_pos").data,
            [self.data.sensor("vacuum_pos").data[2]],
            self.data.qpos[[2, 3, 4]],                 # yaw, shoulder, elbow
            [1.0 if self.attached else 0.0],
        ]).astype(np.float32)
        if self.image_obs:
            return {"agent_pos": obs, "pixels": {"top": self.render()}}
        return obs

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # you can also use free camera
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    # ---- the "grasp": weld constraint on demand ----
    def _try_attach(self):
        grip = self.data.sensor("gripper_pos").data
        grasp = np.array([0, 0, 0])  # recompute grasp site position in world
        grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vacuum_grasp")
        grasp = self.data.site_xpos[grasp_id]
        if np.linalg.norm(grip - grasp) < self.ATTACH_DIST and not self.attached:
            # add a weld connecting vacuum body to the gripper site's body (fore_arm)
            self.attached = True
            self._weld_id = self.model.add_cached() if False else None  # see note below

    # NOTE on attach: the clean way in modern MuJoCo is:
    #   mujoco.mj_equivData / model.eq_type — but dynamically adding constraints
    #   requires rebuilding. The pragmatic sim trick everyone actually uses:

    def _attach(self):
        """Attach vacuum to arm by zeroing its free joint velocity and
        teleporting it along with the gripper every step (kinematic carry)."""
        self.attached = True
        # cache the vacuum's pose relative to the gripper site
        grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vacuum_grasp")
        self._rel_pos = self.data.site_xpos[grasp_id] - self.data.sensor("gripper_pos").data
        self._rel_quat = self.data.qpos[self._vacuum_qposadr + 3:self._vacuum_qposadr + 7].copy()

    def _carry(self):
        """Called each step while attached: kinematically glue vacuum to gripper."""
        if not self.attached:
            return
        grip = self.data.sensor("gripper_pos").data
        qadr = self._vacuum_qposadr
        self.data.qpos[qadr:qadr + 3] = grip + self._rel_pos
        self.data.qpos[qadr + 3:qadr + 7] = self._rel_quat
        self.data.qvel[self.model.jnt_dofadr[self._vacuum_jid]:
                       self.model.jnt_dofadr[self._vacuum_jid] + 6] = 0

    # ---- core API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.attached = False

        # randomize vacuum and dock positions
        vx, vy = self.np_random.uniform(-1, 1, size=2)
        qadr = self._vacuum_qposadr
        self.data.qpos[qadr:qadr + 3] = [vx, vy, 0.06]
        self.data.qpos[qadr + 3] = 1  # unit quat w
        self.model.body_pos[self.model.body("dock").id, :2] = self.np_random.uniform(-1.5, 1.5, size=2)

        self.data.qpos[:5] = [0, 0, 0, 0.3, 0.5]   # base at origin, arm in ready pose
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.data.ctrl[:] = [
            a[0] * 0.5,   # base vx  (m/s)
            a[1] * 0.5,   # base vy
            a[2] * np.pi, # yaw
            a[3] * 1.0,   # shoulder
            a[4] * 1.5,   # elbow
        ]

        # stage 1: try to attach if gripper is at the grasp site
        grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vacuum_grasp")
        grip_pos = self.data.sensor("gripper_pos").data
        if not self.attached and np.linalg.norm(grip_pos - self.data.site_xpos[grasp_id]) < self.ATTACH_DIST:
            self._attach()

        mujoco.mj_step(self.model, self.data)
        self._carry()   # glue the vacuum to the arm after stepping
        mujoco.mj_forward(self.model, self.data)  # refresh sensor/xpos data after manual edits

        self._steps += 1
        robot_xy = self.data.sensor("robot_pos").data[:2]
        vacuum_xy = self.data.sensor("vacuum_pos").data[:2]
        dock_xy = self.data.sensor("dock_pos").data[:2]
        grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vacuum_grasp")
        gripper_to_vacuum = np.linalg.norm(
            self.data.sensor("gripper_pos").data - self.data.site_xpos[grasp_id])

        # ---- multi-stage reward ----
        reward = 0.0
        if not self.attached:
            # stage 1: get the gripper to the vacuum's grasp site
            reward += 0.3 * (1.0 - min(gripper_to_vacuum / 2.0, 1.0))
            reward -= 0.005                            # time penalty
        else:
            # stage 2: carry vacuum to dock
            dist = np.linalg.norm(vacuum_xy - dock_xy)
            reward += 0.5 * (1.0 - min(dist / 3.0, 1.0))
            reward -= 0.002

        success = self.attached and np.linalg.norm(vacuum_xy - dock_xy) < self.DOCK_TOL
        if success:
            reward += 20.0

        terminated = success
        truncated = self._steps >= self.MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, {
            "success": success, "attached": self.attached, "stage": 2 if self.attached else 1
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


register(id="FetchVacuum-v0", entry_point=FetchVacuumEnv)
```

(Heads-up: I left a `_try_attach` stub with a dead branch in there deliberately at first draft — the **kinematic-carry** approach in `_attach`/`_carry` is the actual implementation. If you prefer physics-true attachment, the proper route is defining an `<equality><weld .../></equality>` in the XML with `active="0"` and flipping `model.eq_active0[id]` at runtime — worth knowing, but the kinematic glue is what most fast-iteration sim papers actually ship.)

### 2.3 Scripted sanity check (again: watch before you train)

```python
import gymnasium as gym, numpy as np
from gymnasium.wrappers import RecordVideo

env = gym.make("FetchVacuum-v0", render_mode="rgb_array")
env = RecordVideo(env, "videos", episode_trigger=lambda e: True, name_prefix="vacuum")
obs, _ = env.reset(seed=3)

# crude scripted controller: drive toward vacuum, reach out, then drive to dock
vacuum_xy = obs[2:4]
dock_xy   = obs[4:6]
for t in range(500):
    if not env.unwrapped.attached:
        # point base at vacuum, arm forward
        target = vacuum_xy - obs[0:2]
        action = np.array([np.clip(target[0], -1, 1), np.clip(target[1], -1, 1),
                           0.0, 0.5, 0.5], dtype=np.float32)
    else:
        target = dock_xy - obs[0:2]
        action = np.array([np.clip(target[0], -1, 1), np.clip(target[1], -1, 1),
                           0.0, 0.3, 0.3], dtype=np.float32)
    obs, r, term, trunc, info = env.step(action)
    if term:
        print(f"success at step {t}: {info}")
        break
env.close()
```

Note the scripted controller reads the **obs** (not simulator internals) to decide actions — that's a free test that your observation space is actually sufficient to solve the task. If your script can't solve it from the obs, no learned policy will.

---

## 3. Where to go from here (in rough order of effort)

1. **Cameras**: add `<camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>` in the XML and render with `renderer.update_scene(data, camera="top")` — now `image_obs=True` gives you the `pixels` dict your VLA loop from last time expects verbatim.
2. **Real gripper**: replace the kinematic carry with two box finger geoms + slide joints, and attach on contact force + finger-closed condition. Contact-rich, but that's day-one real grasping.
3. **Curriculum**: randomize dish/vacuum spawn ranges in `reset` via a `difficulty` kwarg — gym registration supports `env_kwargs`, and easier starts + widening ranges is the cheapest performance boost in sim robotics.
4. **Humanoid-lite**: if you want to stay honest to the "walking" fantasy, swap the velocity-controlled slide base for a 4-legged `slide` + `hinge` mess or grab a premade walker (Gymnasium has `Ant-v5`, etc.) and mount the arm on top. Walking while carrying is genuinely hard — great thesis fodder, terrible first project.
5. **Plug in your VLA loop**: both envs expose the `pixels + agent_pos + task string` shape, so the §3 loop from my previous message drops in with only the `task` text changed (`"push the dish into the sink"` / `"fetch the vacuum and bring it to the dock"`). Obviously a base-pretrained VLA won't zero-shot these — you'd fine-tune on teleop/scripted demonstrations recorded with LeRobot's dataset format — but the plumbing is identical, which is the point.

The two biggest debugging lifelines, one more time: **`mujoco.viewer.launch(model, data)`** for poking at the scene interactively, and **always watch a scripted episode video before training**. Want me to sketch the LeRobot dataset recording + SmolVLA fine-tune pipeline for one of these next?




