import re
import numpy as np

SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
SMPL22_HEAD = 15
SMPL22_LHAND = 20
SMPL22_RHAND = 21

EXPI_JOINTS = ("fhead", "lhead", "rhead", "back", "lshoulder", "rshoulder",
               "lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip",
               "lknee", "rknee", "lheel", "rheel", "ltoes", "rtoes")
EXPI_EDGES = ((0, 1), (0, 2), (0, 3), (3, 4), (4, 6), (6, 8), (3, 5), (5, 7),
              (7, 9), (3, 10), (10, 12), (12, 14), (14, 16), (3, 11), (11, 13),
              (13, 15), (15, 17))


def edges_from_parents(parents):
    return [(int(p), j) for j, p in enumerate(parents) if int(p) >= 0]


def smpl22_edges():
    return edges_from_parents(SMPL22_PARENTS)


def expi_edges():
    return [tuple(e) for e in EXPI_EDGES]


def parse_bvh_named(bvh_path):
    names = []
    parents = []
    stack = []
    cur = -1
    pending = -1
    with open(bvh_path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("ROOT") or line.startswith("JOINT"):
                names.append(line.split()[1])
                parents.append(cur)
                pending = len(names) - 1
            elif line.startswith("End Site"):
                names.append(names[cur] + "End")
                parents.append(cur)
                pending = len(names) - 1
            elif line.startswith("{"):
                stack.append(cur)
                cur = pending
            elif line.startswith("}"):
                cur = stack.pop()
            elif line.upper().startswith("MOTION"):
                break
    return names, parents


def worldpos_joint_names(csv_path):
    with open(csv_path) as f:
        header = f.readline().strip()
    names = []
    for col in re.split(r"[,\s]+", header):
        if not col or col.lower() == "time":
            continue
        base = col.rsplit(".", 1)[0]
        if not names or names[-1] != base:
            names.append(base)
    return names


def remocap_skeleton(bvh_path, csv_path):
    bvh_names, bvh_parents = parse_bvh_named(bvh_path)
    name_to_parent = {
        nm: (bvh_names[p] if p >= 0 else None)
        for nm, p in zip(bvh_names, bvh_parents)
    }
    names = worldpos_joint_names(csv_path)
    idx = {nm: i for i, nm in enumerate(names)}
    parents = []
    for nm in names:
        pn = name_to_parent.get(nm)
        parents.append(idx[pn] if pn in idx else -1)
    return names, parents


def read_worldpos_csv(csv_path, n_joints, scale=1.0):
    import numpy as np
    rows = []
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in re.split(r"[,\s]+", line) if x != ""]
            rows.append(vals)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[1] > n_joints * 3:
        arr = arr[:, -n_joints * 3 :]
    return (arr.reshape(arr.shape[0], n_joints, 3) * scale)


def load_bvh(bvh_path):
    names, parents, offsets, channels = [], [], [], []
    stack, cur, pending = [], -1, -1
    with open(bvh_path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if t.startswith("ROOT") or t.startswith("JOINT"):
            names.append(t.split()[1]); parents.append(cur)
            offsets.append((0.0, 0.0, 0.0)); channels.append([])
            pending = len(names) - 1
        elif t.startswith("End Site"):
            names.append(names[cur] + "End"); parents.append(cur)
            offsets.append((0.0, 0.0, 0.0)); channels.append([])
            pending = len(names) - 1
        elif t.startswith("OFFSET"):
            offsets[pending] = tuple(float(x) for x in t.split()[1:4])
        elif t.startswith("CHANNELS"):
            p = t.split(); channels[pending] = p[2:2 + int(p[1])]
        elif t.startswith("{"):
            stack.append(cur); cur = pending
        elif t.startswith("}"):
            cur = stack.pop()
        elif t.upper().startswith("MOTION"):
            break
        i += 1
    while i < len(lines) and not lines[i].strip().startswith("Frame Time"):
        i += 1
    frame_time = float(lines[i].split()[-1]) if i < len(lines) else 0.0
    rows = []
    for line in lines[i + 1:]:
        line = line.strip()
        if line:
            rows.append([float(x) for x in re.split(r"[,\s]+", line) if x != ""])
    frames = np.asarray(rows, dtype=np.float64)
    return names, parents, np.asarray(offsets, np.float64), channels, frames, frame_time


def _axis_rot(axis, deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bvh_fk_world(names, parents, offsets, channels, frames, scale=1.0):
    T, J = frames.shape[0], len(names)
    pos = np.zeros((T, J, 3), np.float32)
    for t in range(T):
        vals = frames[t]; k = 0
        gR = [None] * J; gP = [None] * J
        for j in range(J):
            loc = np.eye(3); trans = np.zeros(3)
            for cn in channels[j]:
                v = vals[k]; k += 1
                if cn.endswith("position"):
                    trans["XYZ".index(cn[0])] = v
                else:
                    loc = loc @ _axis_rot(cn[0], v)
            p = parents[j]
            if p < 0:
                gR[j] = loc; gP[j] = offsets[j] + trans
            else:
                gP[j] = gP[p] + gR[p] @ offsets[j]
                gR[j] = gR[p] @ loc
        pos[t] = np.asarray(gP)
    return pos * scale
