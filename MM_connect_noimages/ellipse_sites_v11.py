### This script outputs a list of 0s and 1s that are interpretable by metamorph. As input, it takes an image width in pixels (w), image height in pixels (h), pixel size in microns (p), radius of the well (r), whether you DONT want to use the ellipse thing (s), and the name of the output file (f)
import math
import numpy as np
import sys
import optparse
import pandas as pd

python_script = True

# import options
if python_script == True:
    p = optparse.OptionParser()
    p.add_option('--width', '-w', default=870, help='image width in pixels (integer)')
    p.add_option('--height', '-t', default = 768, help='image height in pixels (integer)')
    p.add_option('--pixelsize', '-p', default = 1.3, help='pixel size in um (float)')
    p.add_option('--radius', '-r', default = 15000, help='well radius in um (float)')
    p.add_option('--square', '-s', default = False, help='assume square site map (not recommended), default is False (boolean)')
    p.add_option('--filename', '-f', default = 'sites', help='basename of output files')
    options, arguments = p.parse_args()

    image_width_pixels = int(options.width)
    image_height_pixels = int(options.height)
    um_per_pixel = float(options.pixelsize)
    well_radius_um = float(options.radius)
    square = options.square
else:
    image_width_pixels = 870
    image_height_pixels = 768
    um_per_pixel = 1.3
    well_radius_um = 15000
    square = False

## if the image size is not square, use rect.image = 1. This will stretch the circle into an ellipse by a factor equal to the difference between height and width
# e.g. if width > height, stretch circle in y direction into an ellipse, then use the width as your pixel size
# this function takes the following inputs:
    # image_width_pixels: image width in pixels, as measured by scope
    # image_height_pixels: image height in pixels, as measured by scope
    # um_per_pixel: um dimension of 1 square pixel, as measured by scope
    # well_radius_um: radius of the well, as reported by manufacturer
    # square: if true, site map will be assumed to be square. Only do this if image_width and image_height are equal
# returns:
    # a list with:
        # first element = site map (1s and 0s)
        # second element = dataframe containing site and x y coordinates, with center of well as reference
def v1(image_width_pixels, image_height_pixels, um_per_pixel, well_radius_um, square = False):
    # convert types
    image_width_pixels = int(image_width_pixels)
    image_height_pixels = int(image_height_pixels)
    um_per_pixel = float(um_per_pixel)
    well_radius_um = float(well_radius_um)

    if square:
        image_dim_um = ((image_width_pixels+image_height_pixels)/2)*um_per_pixel
        x_rad = well_radius_um
        y_rad = well_radius_um
    else:
        image_dim_um = image_width_pixels*um_per_pixel
        # note that x_rad and y_rad are in site units
        x_rad = well_radius_um / image_dim_um
        y_rad = x_rad * (image_width_pixels/image_height_pixels)

    ## calculate dimensions and create empty matrix

    # determine site grid dimensions.add an extra site to "center" each point on a site square.
    site_dim = int(math.ceil(max(x_rad, y_rad)*2 + 1))

    # initialize the site matrix
    site_matrix = [[0] * site_dim for i in range(site_dim)]

    # define the function that gives the radius at a given angle of an ellipse
    def ab_calc_rad(theta, a, b):
        return(a*b/(np.sqrt((a**2)*((math.sin(theta))**2)+(b**2)*((math.cos(theta))**2))))

    ## for each site, check whether the distance of that site to the center of the ellipse is < or = to well_radius_sites. If so, change to a 1
    ## going to assume that the center of the circle = 0,0 in cartesian coordinates. Thus, the radius of the circle in sites could
    ## initialize row counter
    row_counter = 0
    ## initialize "real_sites" i.e. sites that will actually be imaged and denoated as "1"
    real_site_counter = 0
    ## initialize output for pandas dataframe
    output = []
    ## iterate through each site row
    for site_row in site_matrix:
        # within each row, initialize site counter
        site_counter = 0
        # iterate through potential sites within each row
        for site in site_row:
            # center site coordinates onto a cartesian plane with the circle at the center
            mod_site_coordinates = [float(site_counter) - x_rad, float(row_counter) - y_rad]
            # find theta, then use xy_calc_rad to find the radius distance at that theta
            theta_from_center = math.atan(mod_site_coordinates[1]/mod_site_coordinates[0])
            distance_from_center = np.sqrt(((mod_site_coordinates[0])**2) + ((mod_site_coordinates[1])**2))
            # if the distance from the center smaller than the distance from the center to the outside of the well, this is a site we will keep
            if distance_from_center <= ab_calc_rad(theta_from_center, x_rad, y_rad):
                # add one to the real site counter
                real_site_counter = real_site_counter + 1
                # designate this site on the site matrix as the original site number
                site_matrix[row_counter][site_counter] = real_site_counter
            # move to the next site
            site_counter = site_counter + 1
            # define the site output
            site_output = [real_site_counter, mod_site_coordinates[0], mod_site_coordinates[1], theta_from_center, distance_from_center, ab_calc_rad(theta_from_center, x_rad, y_rad), distance_from_center <= ab_calc_rad(theta_from_center, x_rad, y_rad)]
            # define the site output's index
            site_index = ['site', 'x_cart', 'y_cart', 'theta_polr', 'r_polr', 'dist_center', 'keep_this_site']
            # append the site output to the list of series that will be used to make a dataframe
            output.append(pd.Series(site_output, index = site_index))
        row_counter = row_counter + 1



    ## metamorph reads by column, not by line. So, this needs to be transposed to be readable by metamorph
    new_list_transposed = list(map(list, zip(*site_matrix)))

    ## to map these sites back to the original list, iterate through each row and column, and make a 'site', 'mm_site' dataframe
    mm_site_counter = 0
    site_mapper = []
    for row in new_list_transposed:
        for item in row:
            if item > 0:
                mm_site_counter += 1
                site_mapper.append(pd.Series([item, mm_site_counter], index = ['site', 'mm_site']))
    site_mapper = pd.DataFrame(site_mapper)

    ## now, filter original output
    output = pd.DataFrame(output)
    output = output[output['keep_this_site'] == True]
    output = output.merge(site_mapper, how = 'left', on = 'site')

    return([new_list_transposed, site_matrix, output])

## apply this function to the inputs
final_output = v1(image_width_pixels, image_height_pixels, um_per_pixel, well_radius_um, square = False)

## write site list out to new file
line_counter = 0
outfile = open(options.filename + '.txt', 'w')
for line in final_output[0]:
    for site in line:
        if site != 0:
            site = 1
        outfile.write(str(site) + ' ')
outfile.close()

## write a file that allows for visualization
outfile = open(options.filename + '_readable.txt', 'w')
for line in final_output[1]:
    for site in line:
        if site != 0:
            site = 1
        outfile.write(str(site) + ' ')
    outfile.write('\n')
outfile.close()

print('Dimensions of site map should be %s' % str(len(final_output[1][0])))

## write a file that outputs the coordinates of everything
final_output[2].to_csv('%s_site-coordinates.csv' % options.filename)
