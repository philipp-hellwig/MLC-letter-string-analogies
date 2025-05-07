import time
import math


def asMinutes(s): 
    # convert seconds to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, proportion):
    # prints time elapsed and estimated time remaining
    #
    # Input 
    #  since : previous time
    #  proportion : amount of training complete
    now = time.time()
    s = now - since
    es = s / (proportion)
    rs = es - s
    return f'{asMinutes(s)} (- {asMinutes(rs)} left, {proportion*100:.1f}% completed)'
