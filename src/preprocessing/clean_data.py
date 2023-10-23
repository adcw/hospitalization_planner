def strip_data(data_raw_path: str, data_stripped_path: str):
    with open(data_raw_path, "r") as file_in:
        lines = file_in.readlines()
        stripped_lines = [line.strip() + "\n" for line in lines]

    with open(data_stripped_path, "w+") as file_out:
        file_out.writelines(stripped_lines)


def create_header_constants(data_path: str, colnames_path: str):
    with open(data_path, "r") as file:
        headers: list[str] = file.readline().strip().split(",")

    constants = [f"{name.upper()} = '{name}'\n" for name in headers]

    with open(colnames_path, "w+") as file:
        file.writelines(constants)


if __name__ == '__main__':
    create_header_constants("../../data/input.csv", '../../data/colnames.py')
