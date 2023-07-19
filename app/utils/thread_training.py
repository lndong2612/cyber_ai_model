import threading


class Training:
    """
    Tạo luồng mới để đào tạo mô hình
    """

    def __init__(self):
        pass

    def check_thread_by_name(self, name):
        threads = threading.enumerate()
        for thread in threads:
            if thread.name == name:
                return True
        return False

    def get_thread_by_name(self, name):
        threads = threading.enumerate()
        for thread in threads:
            if thread.name == name:
                return thread

    def start_train(self, id_model, train_func, args):
        print("start training", flush=True)
        thread = threading.Thread(
            target=train_func,
            args=args,
            name=id_model)
        thread.start()

    def stop_train(self, name):
        t = self.get_thread_by_name(name)
        t.do_run = False
