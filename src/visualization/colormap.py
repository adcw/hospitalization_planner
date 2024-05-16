from matplotlib.cm import get_cmap


def get_dtreeviz_colors(n_colors: int, cmap_name: str = 'viridis'):
    colors = get_cmap(cmap_name, n_colors)([i for i in range(n_colors)])

    hex_colors = []
    for i in range(n_colors):
        rgba = colors[i]
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        hex_colors.append(hex_color)

    color_array = [None for _ in range(n_colors + 1)]
    color_array[n_colors] = hex_colors

    return color_array


if __name__ == '__main__':
    colors_list = get_dtreeviz_colors(4)

    pass
