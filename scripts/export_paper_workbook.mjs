// Optional presentation export. Scientific results and reproduction use Python/CSV.
// Requires @oai/artifact-tool; see task6 documentation for the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [dataPath, outputPath, reviewDir] = process.argv.slice(2);
if (!reviewDir) throw new Error('Usage: node export_paper_workbook.mjs workbook_data.json output.xlsx review-directory');
const data = JSON.parse(await fs.readFile(dataPath, 'utf8'));
const wb = Workbook.create();
const labels = {
  claim:'指标 / 结论',value:'数值',unit:'单位',interpretation:'解释边界',repeat:'重复编号',
  method:'方法',raw_rating_mae:'原始分MAE',raw_recall_5:'个人5分精确召回',raw_recall_6:'个人6分精确召回',
  metric:'指标',target:'感官目标',mean_gain:'平均改善',lower:'99%区间下限',upper:'99%区间上限',
  rank:'名次',field:'化学字段',chemical_name:'化合物名称',bootstrap_frequency:'重训入选频率',
  target_passes_BY_0_05:'目标通过BY 0.05',ranking_score:'平均绝对折外SHAP',
  predictive_support_status:'历史预测证据状态',mean:'重复均值',sample_sd:'重复间样本SD',
  beer:'啤酒编号',n_historically_eligible:'历史合格路径点数',n_pass_current:'当前完整通过点数',
  selected:'当前是否入选',actual_after_measured:'是否有迁移后实测',
};
await fs.mkdir(path.dirname(outputPath), {recursive:true});
await fs.mkdir(reviewDir, {recursive:true});
const reviews = [];
for (const [i, view] of data.sheets.entries()) {
  const sheet = wb.worksheets.add(view.name);
  sheet.showGridLines = false;
  sheet.tabColor = i === 0 ? '#687689' : '#273D55';
  const rows = [view.headers.map(h=>labels[h] ?? h), ...view.rows];
  const area = sheet.getRangeByIndexes(4, 0, rows.length, view.headers.length);
  area.values = rows;
  area.format.font = {name:'Arial', size:10, color:'#172A3A'};
  area.format.rowHeight = 25;
  area.format.verticalAlignment = 'center';
  sheet.getRange('A2').values = [[view.title]];
  sheet.getRange('A2').format.font = {name:'Arial',size:14,bold:true,color:'#172A3A'};
  sheet.getRange('A2').format.rowHeight = 30;
  sheet.getRange('A3').values = [[view.note]];
  sheet.getRange('A3').format.font = {name:'Arial',size:10,color:'#536678'};
  sheet.getRange('A3').format.rowHeight = 29;
  sheet.getRangeByIndexes(1,0,1,view.headers.length).format.borders = {
    bottom:{style:'thin',color:'#A7B7C5'},
  };
  for (let c=0; c<view.headers.length; c++) {
    const column = sheet.getRangeByIndexes(4,c,rows.length,1);
    column.format.columnWidth = view.widths[c];
    const numeric = view.rows.some(row=>typeof row[c] === 'number');
    if (numeric) {
      const body = sheet.getRangeByIndexes(5,c,view.rows.length,1);
      body.setNumberFormat(['repeat','rank','beer','n_historically_eligible','n_pass_current'].includes(view.headers[c]) ? '0' : '0.000000');
      body.format.horizontalAlignment = 'right';
    }
  }
  const header = sheet.getRangeByIndexes(4,0,1,view.headers.length);
  header.format.fill = '#273D55';
  header.format.font = {name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  header.format.horizontalAlignment = 'center';
  header.format.rowHeight = 31;
  header.format.borders = {insideVertical:{style:'thin',color:'#FFFFFF'}};
  for (let r=1; r<rows.length; r++) {
    if (r%2 === 0) sheet.getRangeByIndexes(4+r,0,1,view.headers.length).format.fill = '#F1F5F8';
  }
  if (i === 0) {
    sheet.getRangeByIndexes(5,3,view.rows.length,1).format.wrapText = true;
    sheet.getRangeByIndexes(5,0,view.rows.length,view.headers.length).format.rowHeight = 40;
    // Mixed count, score and rate units are explicit in column C.
    for (const [r,row] of view.rows.entries()) {
      sheet.getRangeByIndexes(5+r,1,1,1).setNumberFormat(Number.isInteger(row[1]) ? '0' : '0.000000');
    }
  }
  sheet.freezePanes.freezeRows(5);
  const exported = area.values;
  if (JSON.stringify(exported) !== JSON.stringify(rows)) throw new Error('Cell mismatch: '+view.name);
  reviews.push({sheet:view.name,rows:view.rows.length,columns:view.headers.length,cell_values_match:true});
}
wb.recalculate();
for (const [i, view] of data.sheets.entries()) {
  const preview = await wb.render({sheetName:view.name,autoCrop:'all',scale:1,format:'png'});
  await fs.writeFile(path.join(reviewDir,`${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
  console.log('Rendered '+view.name);
}
const inspection = await wb.inspect({kind:'region',sheetId:data.sheets[0].name,range:'A5:D9',maxChars:1800,tableMaxRows:5,tableMaxCols:4});
await fs.writeFile(path.join(reviewDir,'inspection.json'),JSON.stringify(inspection,null,2));
const result = await SpreadsheetFile.exportXlsx(wb);
await result.save(outputPath);
await fs.writeFile(path.join(reviewDir,'authoring_checks.json'),JSON.stringify({static_values:true,sheets:reviews},null,2));
console.log('Exported '+outputPath);
