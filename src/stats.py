class Stats(object):
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.feedback_recognize_ok = 0
        self.feedback_recognize_fail = 0
        self.queue_size = 0
        self.request_times = []

    def add_request_time(self, time):
        self.request_times.append(time)
        if len(self.request_times) > 10000:
            self.request_times.pop(0)

    def get_stats(self):
        request_times = sorted(self.request_times)

        perc50 = perc90 = perc99 = 0
        if len(request_times) > 0:
            percentiles = [
                request_times[int(len(request_times) * i / 100)] for i in range(100)
            ]
            perc50 = percentiles[50]
            perc90 = percentiles[90]
            perc99 = percentiles[99]

        return (
            "Total requests: {}\n"
            "Successful requests: {}\n"
            "Failed requests: {}\n"
            "Successful recognitions (based on feedback): {}\n"
            "Failed recognitions (based on feedback): {}\n"
            "Tasks queue size: {}\n"
            "Request times 50 percentile: {:.2f} sec\n"
            "Request times 90 percentile: {:.2f} sec\n"
            "Request times 99 percentile: {:.2f} sec\n".format(
                self.requests_total,
                self.requests_success,
                self.requests_failed,
                self.feedback_recognize_ok,
                self.feedback_recognize_fail,
                self.queue_size,
                perc50,
                perc90,
                perc99,
            )
        )
