#%%
import xlsxwriter

def write_xlsx(xlsx_path, xlsx_name, data_list, sim_mat):

    workbook = xlsxwriter.Workbook(xlsx_path + xlsx_name)
    worksheet = workbook.add_worksheet()

    for row in range(len(data_list)+1):
        for col in range(len(data_list)+1):
            if row == 0 and col == 0:
                worksheet.write(row, col, '')
            elif row == 0 and col > 0:
                worksheet.write(row, col, data_list[col-1])
            elif col == 0 and row > 0:
                worksheet.write(row, col, data_list[row-1])
            else:
                worksheet.write(row, col, sim_mat[row-1][col-1])

    workbook.close()