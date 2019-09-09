from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from math import pi, sin, cos
import sys, os
from emg.utils import gen_labels
from panda3d.core import Filename


class Animation(ShowBase):
    def __init__(self, recorder, trials):
        ShowBase.__init__(self)

        labels = gen_labels(trials)

        self._recorder = recorder
        self._labels = iter(labels)
        self._current_label = None

        cur_dir = os.path.abspath(sys.path[0])
        cur_dir = Filename.fromOsSpecific(cur_dir).getFullpath()

        self.pandaActor = Actor(cur_dir + "/emg/animation/Hand")
        scale = 10
        self.pandaActor.setScale(scale, scale, scale)
        self.pandaActor.reparentTo(self.render)

        self.taskMgr.add(self.spin_camera_task, "spin_camera_task")
        self.taskMgr.add(self.run_trial, "run_trial_task")

        self.pandaActor.setPos(7.9, 1.5, -14.5)

    def spin_camera_task(self, task):
        angle_deg = 205  # 20 * task.time * 6.0
        theta = 20
        angle_rad = angle_deg * (pi / 180.0)
        theta_rad = theta * (pi / 180.0)
        self.camera.setPos(
            3.5 * sin(angle_rad), -3.5 * cos(angle_rad), -3.5 * sin(theta_rad)
        )
        self.camera.setHpr(angle_deg, theta, 0)

        return task.cont

    def run_trial(self, task):
        if task.time < 2.1:
            return task.cont

        try:
            # get next label and update animation if it is a new one
            label = next(self._labels)
            if self._current_label != label:
                self._current_label = label
                self.pandaActor.play(label.value)

            self._recorder.record_label(label)

            return task.again
        except StopIteration:
            self._recorder.stop_offline_recording()
            base.destroy()
            sys.exit()
            return task.done
