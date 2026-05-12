from re import search
import numpy as np
import pandas as pd
# import helper
import math
from tqdm import tqdm
from scipy import spatial

# Need to better understand this function to better create the 3D arrray at the end
def spacial_correction (X , Y):
    #Determine the spacial frame
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    # Determine the pattern to generate
    split_list = split_by_same_value(Y)
    correct_list = merge_single_elements(split_list)
    true_pattern = [len(row) for row in correct_list]
    len_y = len(true_pattern)
    len_x = max(set(true_pattern), key=true_pattern.count)

    #generate value with linspace
    row_Y = np.linspace(y_min, y_max, len_y)
    row_Y = np.array([round(y, 2) for y in row_Y])

    row_X = np.linspace(x_min, x_max, len_x)
    row_X = np.array([round(x, 2) for x in row_X])

    x, y= [], []
    nx, ny =[], []

    for i in range (len_y):
        x.append(row_X)
        y.append([row_Y[i]]*len_x)

    for i in range (len_x):
        ny.append(row_Y)
        nx.append([row_X[i]]*len_y)

    x,y = np.array(x), np.array(y)
    nx,ny = np.array(nx),np.array(ny)

    X = pd.Series( np.concatenate(x))
    Y = pd.Series(np.concatenate(y))

    pattern_x_arranged = [len_y]*len_x
    pattern_y_arranged = [len_x]*len_y
    return X,Y,x,y,nx,ny,pattern_x_arranged,pattern_y_arranged

def data_correction (measurement, X, Y):
    data = measurement.drop([0, 1], axis=1)
    data.columns = range(data.columns.size)
    data = data.to_numpy()


    Data_X = measurement[0]
    Data_Y = measurement[1]
    Coord_data = np.array([(Data_X[i],Data_Y[i]) for i in  range(Data_Y.size)])
    tree = spatial.KDTree(Coord_data)

    data_array = []

    for i in range (len(X)):
        search_coord = X[i],Y[i]
        index = tree.query([search_coord])
        waveform = data[index[1].item()]
        data_array.append(waveform)

    return pd.DataFrame(data_array)

def check_list_equality(lst):
    """
    Check if all elements in a list are equal and return non-equal indices.

    Args:
        lst: List to check

    Returns:
        tuple: (bool, list) - (True if all equal, list of non-equal indices)
    """
    if not lst:  # Empty list case
        return True, []

    if len(lst) == 1:  # Single element
        return True, []

    ref_element = lst[len(lst)//2]
    non_equal_indices = []

    for i, element in enumerate(lst):
        if element != ref_element:
            non_equal_indices.append(i)

    all_equal = len(non_equal_indices) == 0
    return all_equal, non_equal_indices

def find_closest_index(lst, target):
    """
    Find the index of a target number(s) in a list, or the index of the closest element.

    Args:
        lst: List of numbers
        target: Number or list of numbers to find or find closest to

    Returns:
        int or list: Index of the target or closest element.
                    If target is a single number, returns int.
                    If target is a list, returns list of indices.

    Raises:
        ValueError: If the list is empty or target list is empty
    """
    if not lst:
        raise ValueError("List cannot be empty")

    # Handle single target
    if not isinstance(target, list):
        return _find_single_closest_index(lst, target)

    # Handle list of targets
    if not target:
        raise ValueError("Target list cannot be empty")

    return [_find_single_closest_index(lst, t) for t in target]

def _find_single_closest_index(lst, target):
    """Helper function to find closest index for a single target."""
    # Check if exact match exists
    if target in lst:
        return lst.index(target)

    # Find closest element
    closest_diff = float('inf')
    closest_index = 0

    for i, num in enumerate(lst):
        diff = abs(num - target)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i

    return closest_index

def split_by_same_value(input_list):

    """
    Split a list into sublists where each sublist contains consecutive occurrences of the same value.

    Parameters:
    -----------
    input_list: list
        A list of values

    Returns
    -------
    result : list
        A list of sublists, where each sublist contains consecutive same values
    """
    result = []
    current_sublist = [input_list[0]]

    for i in range(1, len(input_list)):
        # If current value is the same as previous value
        if input_list[i] == input_list[i - 1]:
            current_sublist.append(input_list[i])
        else:
            # If value changed, append current sublist to result and start a new one
            result.append(current_sublist)
            current_sublist = [input_list[i]]

    # Don't forget to add the last sublist
    result.append(current_sublist)

    return result

def merge_single_elements(sublists):
    """
    Process the result of split_by_same_value to merge single-element sublists
    with the neighboring sublist that has the closest value.

    Parameters:
    -----------
        sublists: list
        A list of sublists from split_by_same_value

    Returns
    -------
    result : list
        A list of sublists with single elements merged with their closest neighbor

    """
    if not sublists:
        return []

    # If there's only one sublist, just return it
    if len(sublists) == 1:
        return sublists

    result = []
    i = 0

    while i < len(sublists):
        current = sublists[i]

        # Skip if current sublist is not a single element
        if len(current) > 1:
            result.append(current)
            i += 1
            continue

        # Single element case
        single_value = current[0]
        left_diff = float('inf')
        right_diff = float('inf')

        # Calculate difference with left neighbor if it exists
        if i > 0 and result:  # Make sure result is not empty
            left_value = result[-1][0]  # All elements in a sublist are the same
            left_diff = abs(single_value - left_value)

        # Calculate difference with right neighbor if it exists
        if i < len(sublists) - 1:
            right_value = sublists[i + 1][0]  # All elements in a sublist are the same
            right_diff = abs(single_value - right_value)

        # Merge with the neighbor that has the closest value
        if left_diff <= right_diff and result:  # Prefer left in case of tie
            result[-1].append(result[-1][0])
        elif i < len(sublists) - 1:  # Merge with right
            # We'll handle this in the next iteration by prepending
            sublists[i + 1] = [sublists[i + 1][0]] + sublists[i + 1]
        else:  # No neighbor to merge with
            result.append(current)

        i += 1

    return result

def split_and_merge(input_list):
    """
    Split a list by same values and merge single elements with their closest neighbor.

    Parameters:
    -----------
        input_list : list
        A list of values

    Returns
    -------
    correct_list : list
        A list of sublists, with single elements merged
    pattern  : list
        A list containing the number of elements of each sublists
    """
    split_list = split_by_same_value(input_list)
    correct_list = merge_single_elements(split_list)
    true_pattern = [len(row) for row in correct_list]

    equal, index_nonequal = check_list_equality(true_pattern)
    if not equal:
        for i in index_nonequal:
            correct_list[i].append(correct_list[i][-1])
        new_pattern = [len(row) for row in correct_list]
        return correct_list, true_pattern,index_nonequal,new_pattern
    else:
        return correct_list, true_pattern, [], []

def list_slice(list_raw, pattern, index_modified):
    """
       Split a list into sublists where each sublist are the number of element define by pattern.

       Parameters:
       -----------
       list_raw : list
            A list of values
       pattern : list
            a list of the number of sublists and the number of element inside
       Returns
       -------
       fragmented_list : list
           A list of sublists, where each sublist are divided by pattern
       """
    fragmented_list = []


    index = pattern[0]
    step1=list_raw[0: index]
    step2 = list(step1)
    fragmented_list.append(step2)

    for i in range(1, len(pattern)):
        fragmented_list.append(list(list_raw[index: pattern[i] + index]))
        index += pattern[i]

    for i in index_modified:
        if i<len(pattern):
            fragmented_list[i].insert(0,fragmented_list[i+1][0])
        else:
            fragmented_list[i].insert(0,fragmented_list[i - 1][0])
    return fragmented_list

def virtual_waveform_add(data, X_series, Y_series, X_sliced, Y_sliced,  index_virtual):
    # Convert dataframe to numpy array for faster processing
    data_array = data.to_numpy()

    index2add=[]
    for index in index_virtual:
        x = X_sliced[index][0]
        y = Y_sliced[index][0]
        diff=[]
        for coordinate in range (0,len(X_series)):
            diff.append(abs(abs(x) - abs(X_series[coordinate]))+abs(abs(y) - abs(Y_series[coordinate])))
        index2add.append(diff.index(min(diff)))

    for index in index2add:
        data_array = np.insert(data_array, index, data_array[index] , 0)

    temp_x = []
    temp_y = []
    for list_index in range (0, len(X_sliced)):
        for value_index in range (0, len(X_sliced[list_index])):
            temp_x.append(X_sliced[list_index][value_index])
            temp_y.append(Y_sliced[list_index][value_index])


    # Convert back to dataframe
    return pd.DataFrame(data_array), pd.Series(temp_x), pd.Series(temp_y)

def data_wrap (data, ref, description, fre_axis=[]):
    wrap_data = []
    i = 1
    for line in data:
        wrap_data.append((line, ref, fre_axis, description, f"{i}"))
        i+=1
    return wrap_data


def transpose_xy_data(x, y):
    """
    Transpose a list containing sublists.

    Parameters:
    -----------
    x : list
       A list of sublist containing floats
    y : list
        A list of sublist containing floats
    Returns
    -------
    new_x : list
        The transposed version of the list x
    new_y : list
        The transposed version of the list y
    """
    new_y = []
    new_x = []

    for i in range(0, len(y[-1])):
        temp_y = []
        temp_x = []

        for j in range(0, len(y)):
            temp_y.append(y[j][i])
            temp_x.append(x[j][i])
        new_y.append(temp_y)
        new_x.append(temp_x)

    npattern = [len(row) for row in new_y]

    return new_x, new_y, npattern

def data_Bscan (data, pattern, Sindex):
    """
       Provide the correct data for the specific x or y constant index given

       Parameters:
       -----------
       data : pandas.DataFrame
          A DataFrame that contains the waveform measurement on each point
       pattern : list
            A list containing the number of elements of each sublists
       Sindex : int
            The index selected in the x or y constant list
       Returns
       -------
       data.loc[start_index:end_index] : pandas.DataFrame
           The data selected for the specific x or y constant index
       """
    start_index = 0
    for i in range(0, Sindex):
        start_index += pattern[i]
    end_index = start_index + pattern[Sindex] - 1
    return data.loc[start_index:end_index]

def ArrangedData_Xconstant (refpattern, pattern, data):
    """
           Provide the correct data for the specific x or y constant index given

           Parameters:
           -----------
           refpattern : list
              A list containing the number of elements of each sublists
           pattern : list
              A list containing the number of elements of each sublists
           data : pandas.DataFrame
                The index selected in the x or y constant list
           Returns
           -------
           data.reindex(index=arranged).reset_index(drop=True).drop([0, 1],1) : pandas.DataFrame
               The rearranged data for an X constant
           """
    """indices_to_select = []
    arranged = []

    for Sindex in range(0,len(refpattern)):
        for i in range(0, refpattern[Sindex]):
            ind = Sindex + i * (pattern[1])
            indices_to_select.append(ind)
    arranged = arranged + list(indices_to_select)

    arranged_data = data.reindex(index=arranged).reset_index(drop=True)
    arranged_data = data.copy()
    arranged_data.columns = range(arranged_data.columns.size)"""

    # Convert dataframe to numpy array for faster processing
    data_array = data.to_numpy()

    # Pre-allocate the result array with the same shape
    arranged_array = np.zeros_like(data_array)

    indices_to_select = []
    for Sindex in range(0, len(refpattern)):
        for i in range(0, refpattern[Sindex]):
            ind = Sindex + i * (pattern[1])
            if ind != sum(pattern):
                indices_to_select.append(ind)
            else:
                indices_to_select.append(pattern[1])

    for index in range(0, len(indices_to_select)):
       arranged_array[index]=data_array[indices_to_select[index]]

    # Convert back to dataframe with original index and columns
    arranged_data = data.copy()
    arranged_data.loc[:, :] = arranged_array

    return arranged_data

def angle_slice_coord(X, Y, angle_degrees, p, treshold=0.1):
    """
    Sweep a rectangle with an angled line

    Args:
        width: rectangle width
        height: rectangle height
        angle_degrees: angle in degrees (0=horizontal, 90=vertical)
        num_steps: number of sweep positions
    """
    theta = math.radians(angle_degrees)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # Find sweep range by projecting corners
    minX, minY = np.min(X), np.min(Y)
    maxX, maxY = np.max(X), np.max(Y)

    # Determine the step
    x_step = round(np.median(np.diff(X)), 2)
    row_y = []
    for y in Y:
        if y not in row_y:
            row_y.append(y)
    y_step = round(np.median(np.diff(row_y)), 2)

    corners = [(minX, minY), (maxX, minY), (minX, maxY), (maxX, maxY)]
    projections = [-sin_theta * x + cos_theta * y for x, y in corners]
    t_min = min(projections)
    t_max = max(projections)

    # Generate sweep positions
    t = t_min + p * (t_max - t_min)

    # Classify each grid cell
    coord_slice = []
    for y in row_y:
        for x in X:
            # Use cell center
            d = abs(-sin_theta * (x + x_step) + cos_theta * (y + y_step) - t)
            if d<treshold :
                if len(coord_slice)==0:
                    coord_slice.append((x, y))
                if (x,y) not in coord_slice and len(coord_slice)>0:
                    coord_slice.append((x,y))

    return coord_slice

def data_Bscan4Angle (data, X, Y, coord_slice):
    data = data.to_numpy()
    tree = spatial.KDTree(np.array([(X[i], Y[i]) for i in range(Y.size)]))

    data_array = []
    save_index = []
    for i in range(len(coord_slice)):
        index = tree.query([coord_slice[i]])
        data_array.append(data[int(index[1])])
        save_index.append(int(index[1]))

    relative_x = np.linspace(0, np.linalg.norm(np.array(coord_slice[-1]) - np.array(coord_slice[0])), len(data_array))

    return pd.DataFrame(data_array), relative_x, save_index

def negative_data_rm (data):
    """
       Replace all negative values in a DataFrame with zeros

       Parameters:
       data (pandas.DataFrame): Input DataFrame

       Returns:
       pandas.DataFrame: DataFrame with negative values replaced by zeros
       """
    # Create a copy to avoid modifying the original DataFrame
    pos_data = data.copy()

    # Replace negative values with zeros
    # This works for all numeric columns in the DataFrame
    pos_data[pos_data < 0.0015] = 0

    return pos_data
def binarization (data, threshold , method="defined"):
    """
        Transform values in a dataframe based on a threshold:
        - Values below the threshold become 0
        - Values above or equal to the threshold become 1

        Parameters:
        -----------
        data : pandas.DataFrame
            The dataframe to transform
        threshold : float
            The threshold value for the transformation in %
        columns : list, optional
            List of column names to apply the transformation to.
            If None, applies to all numeric columns.

        Returns:
        --------
        pandas.DataFrame
            A copy of the dataframe with transformed values
        """
    # Make a copy to avoid modifying the original dataframe
    binary_data = data.copy()
    if method == "defined":
        # If no columns are specified, use all numeric columns
        columns = binary_data.select_dtypes(include=np.number).columns.tolist()

        # Apply the threshold transformation
        for col in columns:
            binary_data[col] = np.where(binary_data[col] < (binary_data[col].max() * (threshold / 100)), 0, 1)
        return binary_data

    elif method == "adaptive":
        print("Not implemented yet")
        return -1
    else:
        print('Not an existing method, please choose between "defined" and "adaptive"')
        return -1


def max_peak_finder (data):
    max_peak_array = []
    for row_index in range (0, data.shape[0]):
        max_peak_array.append(data.loc[row_index][data.loc[row_index].idxmax()])
    return pd.Series(max_peak_array)

# Maybe remove lot of other unused functions
# def thickness_finder (data, sampling_period, refractive_index, threshold_detection):
#     Thickness_array = []
#     for row_index in range(0, data.shape[0]):
#         Thickness_array.append(helper.Layer_calculator(data.loc[row_index], refractive_index, sampling_period,threshold_detection))

#     return pd.Series(Thickness_array)
