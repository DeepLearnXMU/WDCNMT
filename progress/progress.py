# coding=utf-8
import datetime
import os
import subprocess
import sys
import time
import uuid

import numpy
import theano

import ops.random
from delaytask import Resource, Task, Manager


class Progress:
    def __init__(self, delay_val, iterator, seed):
        """
        :param delay_val:
        :param iterator: TextIterator instance
        :param seed:
        """
        self.epoch = 0
        self.batch_total = 0  # count in training
        self.batch_count = 0  # count in epoch
        self.valid_hist = []  # ((epoch, batch_count), score)
        self.loss_hist = []

        self.best_score = -numpy.inf

        self.seed = seed
        self.delay_val = delay_val
        self.task_manager = None
        if self.delay_val:
            self.task_manager = self._init_task_manager()
        self.iterator = iterator
        # initialize
        numpy.random.seed(seed)
        ops.random.seed(seed)
        # re-shuffle corpus to keep randomness consistency
        self.iterator.reset()

        self.oldname = None
        self.serializer = None

        self.elapse = 0

    def _init_task_manager(self):
        manager = Manager(1)

        for id in xrange(3):
            resource = EvalResource('_')
            manager.register_resource(resource)
        return manager

    def save(self, option, name_template, overwrite):
        name = name_template.format(epoch=self.epoch + 1, batch=self.batch_count)
        self.serializer(name, option, self)
        oldname = self.oldname
        if overwrite and oldname != name:
            if oldname and os.path.exists(oldname):
                os.remove(oldname)
        self._garbage()
        self.oldname = name

    def _garbage(self):
        if self.task_manager:
            self.task_manager.garbage()

    def drop_tasks(self):
        """ drop pending tasks """
        if self.task_manager:
            self.task_manager.drop_tasks()

    def terminate(self):
        """
        interrupt signal handler, avoid zombie processes
        :return:
        """
        if self.delay_val:
            self.task_manager.terminate()
        self.iterator.close()

    def add_valid(self, group, src, ref_stem, external_validation_script, main_entry, option, modelname, bestname,
                  serializer):
        if self.delay_val:
            task = EvalTask(group, src, ref_stem, external_validation_script, main_entry, option, self, modelname,
                            bestname, serializer)
            self.task_manager.register_task(task)
            print '[{}-{}] {}'.format(self.epoch + 1, self.batch_count, self.task_manager)
        else:
            evaluate_bleu(src, ref_stem, external_validation_script, main_entry, option, self, modelname, bestname,
                          serializer)

    def barrier(self):
        if self.delay_val:
            self.task_manager.barrier()
        else:
            return

    def failed(self):
        if self.task_manager:
            return self.task_manager.failed()
        else:
            return False

    def tic(self):
        self.elapse -= time.time()

    def toc(self):
        self.elapse += time.time()

    def __getstate__(self):
        d = self.__dict__.copy()
        d['random_state'] = (ops.random.get_state(), numpy.random.get_state())
        del d['serializer']

        return d

    def __setstate__(self, state):
        th_state, np_state = state['random_state']
        del state['random_state']
        self.__dict__.update(state)

        ops.random.set_state(th_state)
        numpy.random.set_state(np_state)

        if self.delay_val:
            for task in self.task_manager.tasks:
                task.progress = self


class EvalResource(Resource):
    def __init__(self, device, *args, **kwargs):
        Resource.__init__(self, *args, **kwargs)
        self.device = device


class EvalTask(Task):
    def __init__(self, group, src, ref_stem, ext_script, main_entry, option, progress, modelname, bestname,
                 serializer,
                 *args, **kwargs):
        Task.__init__(self, group, *args, **kwargs)
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        savename = os.path.join('tmp', '{}.iter{}-{}.{}.pkl'.format(modelname, progress.epoch, progress.batch_count,
                                                                    uuid.uuid4()))
        # take a snapshot
        serializer(savename, option)

        self.progress = progress

        self.src = src
        self.ref_stem = ref_stem
        self.ext_script = ext_script
        self.savename = savename
        self.bestname = bestname
        self.main_entry = main_entry

        self.epoch = progress.epoch
        self.batch_count = progress.batch_count

        self.elapse = 0

    def run(self, shared_dict):
        Task.run(self, shared_dict)
        tic = time.time()

        env = dict(os.environ)
        env['THEANO_FLAGS'] = 'device={}'.format(self.resource.device)
        cmd = '{ext_script} {entry} {model} {src} {ref_stem}'.format(ext_script=self.ext_script,
                                                                     entry=self.main_entry,
                                                                     model=self.savename, src=self.src,
                                                                     ref_stem=self.ref_stem)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        out, err = p.communicate()
        if p.returncode > 0:
            sys.stderr.write('eval task failed, exit code: %d\n' % p.returncode)
            sys.stderr.write('----- task(%d-%d) error -----\n' % (self.epoch + 1, self.batch_count))
            sys.stderr.write(err)
            sys.stderr.write('----- error -----\n')
            raise RuntimeError('Executing external validation script exit unexpectedly')
        bleu_score = float(out)
        toc = time.time()
        rv = (bleu_score, toc - tic)
        Task._send(self, shared_dict, rv)

    def after(self, rv):
        Task.after(self, rv)
        score = rv[0]
        elapse = rv[1]
        self.elapse = elapse
        self.result = score
        progress = self.progress
        if score > progress.best_score:
            progress.best_score = score
            os.rename(self.savename, self.bestname)

        print_eval(self.epoch, self.batch_count, self.progress, True, score, elapse)

    def garbage(self):
        if os.path.exists(self.savename):
            os.remove(self.savename)

    def __getstate__(self):
        d = Task.__getstate__(self)
        del d['progress']  # needs to be set in progress __setstate__
        return d


def evaluate_bleu(src, ref_stem, ext_script, main_entry, option, progress, modelname, bestname, serializer):
    tic = time.time()
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    savename = os.path.join('tmp', '{}.iter{}-{}.{}.pkl'.format(modelname, progress.epoch, progress.batch_count,
                                                                uuid.uuid4()))
    # take a snapshot
    serializer(savename, option)
    cmd = '{ext_script} {entry} {model} {src} {ref_stem}'.format(ext_script=ext_script, entry=main_entry,
                                                                 model=savename, src=src, ref_stem=ref_stem)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu_score = float(out)

    if bleu_score > progress.best_score:
        progress.best_score = bleu_score
        os.rename(savename, bestname)
    else:
        os.remove(savename)
    elapse = time.time() - tic

    print_eval(progress.epoch, progress.batch_count, progress, False, bleu_score, elapse)


def print_eval(epoch, batch_count, progress, delay, score, elapse):
    item = ((epoch, batch_count), score)
    progress.valid_hist.append(item)
    print '{}-{} bleu: {:.4f}, {}'.format(epoch + 1, batch_count, score,
                                          datetime.timedelta(seconds=int(elapse)))
    if delay:
        print '[{}-{}] {}'.format(progress.epoch + 1, progress.batch_count, progress.task_manager)

