import rpyc
import pickle

class CellClassifierClient:
    def connect(self, *args, **kwargs):
        self.conn = rpyc.connect(*args,**kwargs)

    def run(self, *args, **kwargs):
        '''
        :param args: See cell_analysis.analyze_image
        :param kwargs: See cell_analysis.analyze_image
        :return: See cell_analysis.analyze_image
        '''
        # Hack to get rpyc to work with numpy arrays
        kwargs['do_pickle'] = True
        return  pickle.loads(self.conn.root.run(*args, **kwargs))

    def write_metadata(self, *args, **kwargs):
        self.conn.root.write_metadata(*args, **kwargs)


