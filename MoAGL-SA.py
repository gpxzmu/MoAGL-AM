""" RUN MoAGL-SA
"""
from train_test import train_test

if __name__ == "__main__":    


    data_folder = 'BRCA'
    view_list = [1,2,3]
    # view_list = [3]
    num_epoch = 2500
    num_class = 5
    lr_c = 1e-4
    lr_e = 1e-3


    train_test(data_folder, view_list, num_class, lr_c, lr_e, num_epoch)


