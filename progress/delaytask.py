import itertools
import multiprocessing
import threading
import time
import traceback


class FakeProcProxy:
    def is_alive(self):
        return False

    def join(self):
        return

    def terminate(self):
        return

    @property
    def exitcode(self):
        return 0


class TaskStatus:
    def __init__(self):
        pass

    @staticmethod
    def wait():
        return 1

    @staticmethod
    def run():
        return 2

    @staticmethod
    def done():
        return 4


class ManagerStatus:
    def __init__(self):
        pass

    @staticmethod
    def default():
        return 1

    @staticmethod
    def watching():
        return 2

    @staticmethod
    def stopped():
        return 4


class Resource:
    def __init__(self, *args, **kwargs):
        self._available = True
        self.id = 0
        pass

    @property
    def available(self):
        return self._available

    def take(self):
        self._available = False
        return self

    def free(self):
        self._available = True
        return self


class Task:
    def __init__(self, group, *args, **kwargs):
        """
        :param group: group identifier
        :param args:
        :param kwargs:
        """
        self.resource = None
        self.group = group
        self.proc = None
        self.result = None
        self.id = 0
        self.status = TaskStatus.wait()

    def before(self):
        self.status = TaskStatus.run()
        self.resource.take()

    def run(self, shared_dict):
        # should save any return value in shared_dict with key=id
        # self._send(shared_dict,rv)
        self._send(shared_dict, None)

    def _send(self, shared_dict, rv):
        shared_dict[self.id] = rv

    def after(self, rv):
        self.status = TaskStatus.done()
        self.result = rv
        self.resource.free()

    def is_alive(self):
        if self.proc:
            return self.proc.is_alive()
        return False

    @property
    def exitcode(self):
        if self.proc:
            return self.proc.exitcode
        else:
            return None

    def join(self):
        if self.proc:
            self.proc.join()

    def terminate(self):
        if self.proc:
            self.proc.terminate()

    def garbage(self):
        pass

    def __getstate__(self):
        d = self.__dict__.copy()

        del d['proc']
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.status == TaskStatus.done():
            self.proc = FakeProcProxy()
        else:
            self.status = TaskStatus.wait()
            if self.resource:
                self.resource.free()
                self.resource = None
            self.proc = None


class Manager:
    def __init__(self, check_interval, *args, **kwargs):
        self.resources = []
        self.tasks = []
        self._pending_tasks = []  # pending output

        manager = multiprocessing.Manager()
        self.shared_dict = manager.dict()  # pass and manage values between manager and tasks

        self.interval = check_interval
        self._status = ManagerStatus.default()
        self._thread = None

        self._thread_flag = None

    def try_start(self):
        if self._status == ManagerStatus.default():
            self._status = ManagerStatus.watching()
            self._watch()

    @property
    def running_tasks(self):
        return [task for task in self.tasks if task.status == TaskStatus.run()]

    @property
    def waiting_tasks(self):
        return [task for task in self.tasks if task.status == TaskStatus.wait()]

    @property
    def done_tasks(self):
        return [task for task in self.tasks if task.status == TaskStatus.done()]

    @property
    def ordered_done_tasks(self):
        tasks = []
        for t1, t2 in itertools.izip(self.tasks[:-1], self.tasks[1:]):
            if t1.status == TaskStatus.done():
                tasks.append(t1)
            if t2.status != TaskStatus.done():
                break
        if len(self.tasks) == 1 and self.tasks[0].status == TaskStatus.done():
            tasks.append(self.tasks[0])
        return tasks

    def register_resource(self, resource):
        resource.id = len(self.resources)
        self.resources.append(resource)

    def register_task(self, task):
        if self._status == ManagerStatus.stopped():
            raise RuntimeError("Illegal task registration: manager is already stopped.")
        task.id = len(self.tasks)
        self.tasks.append(task)
        self.try_start()

    def available_resource(self):
        # one resource at a time
        for resource in self.resources:
            if resource.available:
                yield resource

    def available_task(self):
        for task in self.tasks:
            if task.status == TaskStatus.wait():
                yield task

    def drop_tasks(self):
        """ drop pending tasks """
        self.tasks = [task for task in self.tasks if task not in self.waiting_tasks]

    def free(self):
        multiprocessing.active_children()  # clean up, implicitly join finished procs
        for task in self.tasks:
            if task.status == TaskStatus.run():
                if self._is_done(task):
                    task.after(self.shared_dict[task.id])
                    self._pending_tasks.append(task)

    def _run_task(self, task, resource):
        task.resource = resource
        task.before()
        p = multiprocessing.Process(target=task.run, args=(self.shared_dict,))
        p.daemon = True  # when main job ends, kill all subprocess
        task.proc = p
        p.start()

    def _watch(self):
        try:
            # free unused resources
            self.free()
            # until resources or tasks are exhausted
            for resource, task in itertools.izip(self.available_resource(), self.available_task()):
                self._run_task(task, resource)

            # check again later
            if self._status != ManagerStatus.stopped() or list(self.available_task()):
                self._thread = threading.Timer(self.interval, self._watch)
                self._thread.start()
        except Exception:
            self._thread_flag = -1
            traceback.print_exc()
            raise RuntimeError("Manager timer thread failure")

    def _is_done(self, task):
        """
        whether task just finished
        :param task:
        :return:
        """
        if task.status == TaskStatus.run():
            if not task.is_alive():
                if task.exitcode and task.exitcode > 0:
                    raise RuntimeError("task({}-{}) failed".format(task.group, task.id))
                if task.id in self.shared_dict:
                    return True
        return False

    def collect_done(self):
        self.try_start()
        self.free()
        done = self._pending_tasks
        for task in self.tasks:
            #  status of sequences:
            #   wait wait wait wait
            #   run run run wait
            #   done run run wait
            if task.status == TaskStatus.wait():
                break
            if task.status == TaskStatus.run():
                if self._is_done(task):
                    task.after(self.shared_dict[task.id])
                    done.append(task)
                else:
                    break

        self._pending_tasks = []  # clear

        return sorted(done, key=lambda x: x.id)

    def stop(self):
        self._status = ManagerStatus.stopped()

    def terminate(self):
        """
        interrupt signal handler, avoid zombie processes
        :return:
        """
        self.stop()
        if self._thread:
            self._thread.cancel()
            self._thread.join()
        for task in self.tasks:
            task.terminate()
            task.join()

    def barrier(self):
        self.try_start()
        # blocked
        # wait until all results are retrieved

        self.stop()
        if self._thread:
            self._thread.cancel()
            self._thread.join()
        while list(self.available_task()):  # wait for remaining tasks
            time.sleep(self.interval)

        for task in self.tasks:
            task.join()
        return self.collect_done()

    def garbage(self):
        for task in self.done_tasks:
            task.garbage()

    def failed(self):
        """
        handle timer thread failure outside
        :return:
        """
        return self._thread_flag == -1

    def __str__(self):
        n_run = len(self.running_tasks)
        n_wait = len(self.waiting_tasks)
        n_done = len(self.done_tasks)
        n_total = len(self.tasks)
        return "tasks(R/W/D/T): {}/{}/{}/{}".format(n_run, n_wait, n_done, n_total)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["shared_dict"] = dict(d["shared_dict"])
        del d["_thread"] # _thread will be auto-created when monitor starts
        del d["_thread_flag"]

        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        manager = multiprocessing.Manager()
        self.shared_dict = manager.dict()  # pass and manage values between manager and tasks
        for k, v in state["shared_dict"].iteritems():
            self.shared_dict[k] = v

        for resource in self.resources:
            resource.free()
        self._thread = None
        self._thread_flag = None
        self._status = ManagerStatus.default()

