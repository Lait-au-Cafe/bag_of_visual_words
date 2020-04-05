import inspect

def log(obj, msg: str, end="\n"):
    frame = inspect.currentframe().f_back
    #print(inspect.getframeinfo(frame).function)
    print(f"[{type(obj)} {inspect.getframeinfo(frame).function}] {msg}", end=end)

#log = lambda msg: print(f"[{sys._getframe().f_code.co_name}] {msg}")