import tensorflow as tf
import json
import re


# dummy class to restore constant file
class Constant:
    pass

# dump constant variables into json file
def dump_constants(constants, path):
    data = {}
    for name in dir(constants):
        if re.match(r'^([A-Z]|_|[0-9])+$', name):
            data[name] = getattr(constants, name)
    json_str = json.dumps(data)
    with open(path, 'w') as f:
        f.write(json_str + '\n')

# restore a constant object from json file
def restore_constants(path):
    constants = Constant()
    with open(path, 'r') as f:
        json_obj = json.loads(f.read())
        for key, value in json_obj.items():
            setattr(constants, key, value)
    return constants

class TfBoardLogger:
    def __init__(self, writer):
        self.placeholders = {}
        self.summaries = {}
        self.writer = writer

    def register(self, name, dtype):
        placeholder = tf.placeholder(dtype, [], name=name)
        self.placeholders[name] = placeholder
        self.summaries[name] = tf.summary.scalar(name + '_summary', placeholder)

    def plot(self, name, value, step):
        sess = tf.get_default_session()
        placeholder = self.placeholders[name]
        summary = self.summaries[name]
        out, _ = sess.run(
            [summary, placeholder],
            feed_dict={placeholder: value}
        )
        self.writer.add_summary(out, step)

class JsonLogger:
    def __init__(self, path, overwrite=True):
        self.f = open(path, 'w' if overwrite else 'wa')

    def plot(self, **kwargs):
        json_str = json.dumps(kwargs)
        self.f.write(json_str + '\n')

    def close(self):
        self.f.close()
