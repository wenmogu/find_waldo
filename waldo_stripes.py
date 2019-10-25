import numpy as np

# Method: compute the number of continuous positive/negative pixels
# input: a 1D numpy array of 1s and -1s
# output: a 1D numpy array of positive numbers and negative numbers alternating in order
        # where the absolute value of one number is the length of the streak of the positive/negative pixels
def get_streak_len_array(unit_lst):
    previous = 0
    streak_len_array = []
    streak_len = 0
    for i in range(len(unit_lst)):
        if i == 0:
            previous = unit_lst[i]
        
        if previous == unit_lst[i]:
            streak_len += unit_lst[i]
            if i == len(unit_lst) - 1:
                streak_len_array.append(streak_len)
        else:
            streak_len_array.append(streak_len)
            streak_len = 0
            
        previous = unit_lst[i]
        
    return streak_len_array


# Method: compute the number of continuous positive/negative pixels with approximation
# input: a 1D numpy array of 1s and -1s, a positive integer 
# output: a 1D numpy array of positive numbers and negative numbers alternating in order
        # where the absolute value of one number is the length of the streak of the positive/negative pixels
def get_streak_len_array_with_approx(unit_lst, threshold):
    # in the unit_lst, 1 represents the superior pixel value, -1 is the inferior pixel value
    # the superior pixel value is the one that is well captured in color distillation 
    # in our case e.g. red is the superior pixel value because it is well captured 
    # and white is the inferior pixel value because it might be noise
    # we want to restore the shape outlined by the superior pixel value
    # hence we need to minimize the disturbannce by the inferior pixel value
    # hence we need to remove the 'sparse' inferior pixel value located within the streaks of superior pixel values
    # the definition of sparse is given by the threshold
    previous = 0
    streak_len_array = []
    streak_len = 0
    fast_forward_count = 0
    for i in range(len(unit_lst)):
        if i == 0:
            previous = unit_lst[i]
            
        if fast_forward_count > 0:
            fast_forward_count -= 1
            if i == len(unit_lst) - 1:
                streak_len_array.append(streak_len)
            continue
        
        if previous == unit_lst[i]:
            cur_streak_len = streak_len
            streak_len += unit_lst[i]
            if i == len(unit_lst) - 1:
                streak_len_array.append(streak_len)
            previous = unit_lst[i]
        else:
            if streak_len > 0:
                # superior pixel streak ending
                cur_streak_len = streak_len
                # we need to look ahead the threshold number of pixels to see if theres any superior pixel
                next_few_number = min(threshold, len(unit_lst) - 1 - i)
                next_few_pixels = unit_lst[i : i+1+next_few_number]
                # if there is we will continue the superior pixel streak from there
                indice_of_next_superior_pixel, = np.where(next_few_pixels == 1)
                if len(indice_of_next_superior_pixel) != 0:
                    next_superior = max(indice_of_next_superior_pixel)
                    streak_len += 1 + next_superior
                    fast_forward_count = next_superior
                    previous = abs(unit_lst[i])
                # if not we end the superior pixel streak here and start a new inferior pixel streak
                else:
                    streak_len_array.append(cur_streak_len)
                    streak_len = 0
                    streak_len += unit_lst[i]
                    previous = unit_lst[i]
            elif streak_len < 0:
                # inferior pixel streak ending
                # we will start a new superior pixel streak
                streak_len_array.append(streak_len)
                streak_len = 0
                streak_len += unit_lst[i]
                previous = unit_lst[i]
            
    return streak_len_array


# Method: Given a 1D array, find the starting position and the ending position of an arithmatic sequence with the next equal to 1+previous. The sequence must contain no less than 3 numbers.
# Input: a 1D array of integers
# Output: a tuple in the form of (Boolean, list). The boolean is True when there is at least one sequence. The list consists of tuples (starting_position_of_sequence_inclusive, ending_position_of_sequence_exclusive)
def is_indice_continuous(lst_of_indice):
    # right now the number of stripes accepted is 3
    is_continuous = False
    continuous_count = 0
    prev = -1
    start_index = -1
    lst_of_start_and_end = []
    for i in range(len(lst_of_indice)):
        if i == 0:
            prev = lst_of_indice[i]
            start_index = i
            continuous_count = 1
            continue
            
        if lst_of_indice[i] - prev == 1:
            # the streak continues
            continuous_count += 1
        else:
            # the streak ends
            if continuous_count >= 3:
                lst_of_start_and_end.append((start_index, i))
                is_continuous = True

            continuous_count = 1
            start_index = i
            
        if i == len(lst_of_indice) - 1:
            if start_index != -1 and continuous_count >= 3:
                lst_of_start_and_end.append((start_index, i))
                is_continuous = True
        
        prev = lst_of_indice[i]
        
    return (is_continuous, lst_of_start_and_end)


def get_suspected_waldo_stripe_region_for_col(red_mask_col, white_mask_col, ratio_range=(1/2, 2), white_pixel_threshold=0.3):
    assert red_mask_col.shape == white_mask_col.shape, "the mask column inputs for white and red masks are of different shapes."
    
    new_red_mask_col = np.zeros(len(red_mask_col))
    new_red_mask_col[red_mask_col == 255] = 1
    new_red_mask_col[red_mask_col == 0] = -1
    
    new_white_mask_col = np.zeros(len(white_mask_col))
    new_white_mask_col[white_mask_col == 255] = 1
    new_white_mask_col[white_mask_col == 0] = 0
    
    streak_array = get_streak_len_array_with_approx(new_red_mask_col, 0)
    ratio_array = np.array([abs(streak_array[i]) / abs(streak_array[i + 1]) for i in range(len(streak_array) - 1)])
    accepted_ratio_indice, = np.where(np.logical_and(ratio_array > ratio_range[0], ratio_array < ratio_range[1]))
    is_accepted_ratio_indice_continuous, lst_of_start_and_end = is_indice_continuous(accepted_ratio_indice)
    revised_lst_of_start_and_end = []
    if is_accepted_ratio_indice_continuous:
        # then check if the negative pixels are white pixels
        for start, end in lst_of_start_and_end:
            is_region_start_with_red = streak_array[start] > 0
            
            offset = 0
            if is_region_start_with_red:
                offset = 1
            
            total_number_of_white_pixels = 0
            total_number_of_non_red_pixels = 0
            while start + offset < end:
                supposedly_white_pixel_region = [np.sum(np.abs(streak_array[:accepted_ratio_indice[start + offset]])), np.sum(np.abs(streak_array[:accepted_ratio_indice[start + offset + 1]]))]
                number_of_white_pixels = np.sum(new_white_mask_col[supposedly_white_pixel_region[0] : supposedly_white_pixel_region[1]])
                number_of_non_red_pixels = supposedly_white_pixel_region[1] - supposedly_white_pixel_region[0]
                total_number_of_white_pixels += number_of_white_pixels
                total_number_of_non_red_pixels += number_of_non_red_pixels
                offset += 2
            if total_number_of_white_pixels / total_number_of_non_red_pixels > white_pixel_threshold:
                revised_lst_of_start_and_end.append((start, end))
        
        
        
        # stripe_region = [[starting_pixel_of_Streak, ending_pixel], [starting_pixel, ending_pixel], [starting_pixel, ending_pixel], ...]
        stripe_region = [[np.sum(np.abs(streak_array[:accepted_ratio_indice[start]])), np.sum(np.abs(streak_array[:accepted_ratio_indice[end]]))] for start, end in revised_lst_of_start_and_end]
        # from here examine the percentage of white pixels in the region
        # if there are white pixels dominating the negative pixels, then it is stripes
        # return (1, start_of_stripe_region, end_of_stripe_region, number_of_stripes, average_width_of_stripes)
        return (True, stripe_region)
    
    return (False, [])
                