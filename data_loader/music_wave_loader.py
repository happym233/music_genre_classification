import os
import re

'''
    load all music filenames of designated genre list
    
    root: str
        the root path of music files
    genre list: list
        the list of genres that wants to be selected
'''


def get_music_file_names(root='original_data/genres_original', genre_list=[]):
    music_name_lists = []
    for genre in genre_list:
        music_name_lists.append(list(map(lambda x: genre + '/' + x,
                                         filter(lambda x: re.match(genre + '.(.*).wav', x),
                                                os.listdir(root + genre)))))
    return music_name_lists
