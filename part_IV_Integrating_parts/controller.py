from TFL_manager import TFL
import matplotlib.pyplot as plt


class Controller:
    def run(self):
        path_images, tfl_manager, first_frame = self.init()

        for i in range(len(path_images)):
            tfl_manager.run_product(path_images[i], i+int(first_frame))

        plt.show(block=True)

    def init(self):
        data = open("play_list.pls", "r+")
        read_lines = data.readlines()
        image_path = [line[:-1] for line in read_lines]
        tfl_manager = TFL(image_path[0])

        return image_path[2:], tfl_manager, image_path[1]


c = Controller()
c.run()
