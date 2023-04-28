import time
import datetime
import calendar


def get_date():
    dt = datetime.datetime.now()
    month = calendar.month_name[dt.month][:3]
    date = f'{month} {str(dt.day).zfill(2)} {str(dt.hour).zfill(2)}:{str(dt.minute).zfill(2)}:{str(dt.second).zfill(2)} {str(dt.year).zfill(4)}'
    return date


class Progress:

    def __init__(self):
        self.time_start = time.time()
    
    def restart(self):
        self.time_start = time.time()

    def nab(self, loop, loops):

        time_finish = time.time()
        times = time_finish - self.time_start
        elap_hour = int(times + 0.5) // 3600
        elap_min = (int(times + 0.5) // 60) % 60
        elap_sec = int(times + 0.5) % 60
        elap = f'{str(elap_hour).zfill(2)}:{str(elap_min).zfill(2)}:{str(elap_sec).zfill(2)}'

        times_per_loop = times / loop
        rem_times = times_per_loop * (loops - loop)
        rem_hour = int(rem_times + 0.5) // 3600
        rem_min = (int(rem_times + 0.5) // 60) % 60
        rem_sec = int(rem_times + 0.5) % 60
        rem = f'{str(rem_hour).zfill(2)}:{str(rem_min).zfill(2)}:{str(rem_sec).zfill(2)}'
            
        dt = datetime.datetime.now()
        month = calendar.month_name[dt.month][:3]
        now_date = f'{month} {str(dt.day).zfill(2)} {str(dt.hour).zfill(2)}:{str(dt.minute).zfill(2)}:{str(dt.second).zfill(2)} {str(dt.year).zfill(4)}'

        fin_dt = dt + datetime.timedelta(seconds=rem_times)
        fin_month = calendar.month_name[fin_dt.month][:3]
        fin_date = f'{fin_month} {str(fin_dt.day).zfill(2)} {str(fin_dt.hour).zfill(2)}:{str(fin_dt.minute).zfill(2)}:{str(fin_dt.second).zfill(2)} {str(fin_dt.year).zfill(4)}'

        return elap, times_per_loop, rem, now_date, fin_date



            