
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy.ndimage import gaussian_filter
import scipy.stats
from sklearn.model_selection import KFold
import scipy.signal

class exercise_1:
    """ 1. exercise"""
    def __init__(self, sample=1000, bins=30, parameters=(1, 0.2)):
        self.samp = sample
        self.bins = bins
        self.mean, self.deviation = parameters
        self.normal_line = np.array([])
        

    def draw(self):
        self.normal_line = np.random.normal(self.mean, self.deviation, self.samp)

    def show(self):
        self.draw()

        fig = plt.figure()

        ax0 = fig.add_subplot(1, 2, 1)
        ax0.hist(self.normal_line, bins=self.bins)
        ax1 = fig.add_subplot(1, 2, 2)
        #x = np.arange(0, 2, 0.01)
        x = np.arange(np.min(self.normal_line), np.max(self.normal_line), 0.01)
        y = 1 / (self.deviation * np.sqrt(2 * np.pi)) * \
                np.exp(-(x - self.mean) ** 2 / (2 * self.deviation ** 2))
        ax1.plot(x, y, color='r')

        plt.show()

class exercise_2:
    def __init__(self):
        self.img = scipy.misc.face(gray=True)

    def draw(self):
        self.img_blured = gaussian_filter(self.img, 3)


        img_1D = self.img_blured.reshape(1024 * 768)

        all_values = sum(img_1D)

        img_1D = img_1D/all_values

        img_1D_cdf = np.cumsum(img_1D)

        random_numbers = np.random.uniform(0,1,100000)

        img_1D_sampled = np.searchsorted(img_1D_cdf,random_numbers)

        """create a new image"""
        self.empty_img = np.zeros_like(self.img)

        empty_img_1D =  self.empty_img.reshape(1024 * 768) 

        empty_img_1D[img_1D_sampled] += 1

        empty_img_1D = empty_img_1D.reshape((768,1024))

        return self.img_blured, empty_img_1D

    def show(self):
        self.draw()
        fig = plt.figure()

        #ax0 = fig.add_subplot(1, 2, 1)
        #plt.imshow(self.img, cmap='gray')

        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(self.img_blured, cmap='gray')

        plt.show()

class exercise_2_rewrite:
    def __init__(self, sample_num, gaussina_win_size=3):
        self.sample_num = sample_num

        # get the gray picture raccoon and blur it
        self.img_blured = np.array(gaussian_filter(scipy.misc.face(gray=True), sigma=gaussina_win_size).copy())
        # accroding to density change this picture to 1D probability mass
        self.img_1D_proba = (self.img_blured / np.sum(self.img_blured)).reshape(self.img_blured.size)
        # compute the 1D cumulative density function array
        self.img_1D_cdf = np.cumsum(self.img_1D_proba)

    def draw_samples(self):
        uniform_distr_array = np.random.uniform(0, 1, self.sample_num)
        corresponding_indices = np.searchsorted(self.img_1D_cdf, uniform_distr_array)

        # creat a new empty image
        img_new = np.zeros_like(self.img_blured)

        # draw points which sampled from density
        img_new_1D = img_new.reshape(img_new.size)
        for i in corresponding_indices:
            img_new_1D[i] = self.img_blured.reshape(self.img_blured.size)[i]

        return img_new, corresponding_indices

    def show(self):
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.imshow(self.img_blured, cmap='gray')
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.imshow(self.draw_samples()[0])

        plt.show()

class exercise_3:
        def __init__(self, sampled_image, parzen_win_size=9):
            self.parzen_window_size = parzen_win_size
            # get sample array
            #self.img_sampled_2D, self.index_sample_array = exercise_2_rewrite(sample_num).draw_samples()
            self.img_sampled_2D = sampled_image
            #self.img_reconstruct = np.empty_like(self.img_sampled_2D)

        def compute_proba(self):

            ## init a parzen window in 2D
            parzen_window_2D = np.ones((self.parzen_window_size, self.parzen_window_size))

            image_output = scipy.signal.convolve2d(self.img_sampled_2D, parzen_window_2D, mode='same')

            image_output = image_output / np.sum(image_output)

            return image_output

        def show(self):
            fig = plt.figure()
            # self.img_reconstruct.reshape(self.img_sampled_2D.size) = self.compute_proba()

            plt.imshow(self.compute_proba())

            plt.show()

class exercise_4:
    def __init__(self, candidate_win_size, sample_img, sample_index, k_fold):
        self.sampled_img = sample_img
        self.k_fold = k_fold
        self.sample_index = sample_index
        self.candidate_win_size = candidate_win_size

    def model_sel(self):
        #candidate_win_size = np.array([3, 9, 20])
        proba_list = []

        for win_size in self.candidate_win_size:
            test_praba = 0
            # split samples into K folds
            kf = KFold(n_splits=self.k_fold)
            for x, y in kf.split(self.sample_index):
                img_1D = self.sampled_img.reshape(self.sampled_img.size).copy()
                testset_indices_in_samples = self.sample_index[y]

                img_1D[testset_indices_in_samples] = 0
                trainning_set_2D = img_1D.reshape(self.sampled_img.shape)

                ex3 = exercise_3(trainning_set_2D, parzen_win_size=win_size)
                trained_img = ex3.compute_proba()
                trained_img_1D = trained_img.reshape(trained_img.size)

                test_praba += np.sum(np.log(trained_img_1D[testset_indices_in_samples] + 1))

            # save test proba in current fold into a list
            proba_list.append(test_praba)

        return np.array(proba_list)

if __name__ == "__main__":
# for testing ex1.1
    #ex1 = exercise_1()
    #ex1.show()
#end

# for testing ex1.2
    #ex2_2 = exercise_2_rewrite(100000)
    #ex2_2.show()
# end

# for testing ex1.3
    #ex2_2 = exercise_2_rewrite(100000)
    #sampled_img, _ = ex2_2.draw_samples()
    #ex3 = exercise_3(sampled_img, parzen_win_size=12)
    #ex3.show()
# end

# for testing ex1.4
    candidate_sample_size = np.array([10000, 50000, 100000, 200000, 400000])
    title_array = [str(x) for x in candidate_sample_size]
    candidate_win_size = np.arange(3, 31, step=3)
    k_fold = 10
    density = []

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for idx, value in enumerate(candidate_sample_size):
        ex2 = exercise_2_rewrite(sample_num=value)
        sample_img, index = ex2.draw_samples()
        #ex3 = exercise_3(sample_img, parzen_win_size=30)
        #ex3.show()
        ex4 = exercise_4(candidate_win_size, sample_img, index, k_fold)
        proba_list = ex4.model_sel()

        best_win_size = candidate_win_size[np.argmax(proba_list)]
        #print("if sample size is ", value, ", then best window size is ", best_win_size)
        density.append(np.max(proba_list))

        ax = fig.add_subplot(2, 3, idx + 1)
        ax.set_title("win_size=" + title_array[idx])
        ax.plot(candidate_win_size, proba_list, '-or')
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title("density of test set in best win size")
    ax.plot(candidate_sample_size, density, '--og')

    plt.show()
# end

    #ex2 = exercise_2()
    #img_blured, sampled_img = ex2.draw()

    #ex3 = exercise_3(sampled_img)
    #img_ex3 = ex3.compute_proba()

    #fig = plt.figure()
    #ax0 = fig.add_subplot(1, 3, 1)
    #ax0.imshow(img_blured, cmap='gray')
    #ax1 = fig.add_subplot(1, 3, 2)
    #ax1.imshow(sampled_img)
    #ax2 = fig.add_subplot(1, 3, 3)
    #ax2.imshow(img_ex3)

    #plt.show()

    #ex2 = exercise_2()
    #sample_img = ex2.draw()
    #ex2.show()

    #ex2_2 = exercise_2_rewrite(100000)
    #ex2_2.show()

    #ex3 = exercise_3(sample_img)
    #ex3.show()

