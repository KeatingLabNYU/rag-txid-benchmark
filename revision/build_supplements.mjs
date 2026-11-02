import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const base = path.dirname(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/, '')));
const matrices = JSON.parse(await fs.readFile(path.join(base, 'supplement_matrices.json'), 'utf8'));
const outDir = path.join(base, 'outputs', '01a18309-3fb9-77e2-aeb1-3dd94283d52a');
await fs.mkdir(outDir, {recursive: true});

for (const [supplement, sheets] of Object.entries(matrices)) {
  const wb = Workbook.create();
  for (const [name, rows] of Object.entries(sheets)) {
    const ws = wb.worksheets.add(name);
    ws.showGridLines = false;
    ws.freezePanes.freezeRows(1);
    const r = ws.getRangeByIndexes(0, 0, rows.length, rows[0].length);
    r.values = rows;
    r.format.font = {name: 'Aptos', size: 10};
    r.format.verticalAlignment = 'center';
    r.format.wrapText = true;
    const header = ws.getRangeByIndexes(0,0,1,rows[0].length);
    header.format = {fill:'#17365D', font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},
                     wrapText:true,verticalAlignment:'center'};
    header.format.rowHeight = 36;
    ws.getRangeByIndexes(1,0,Math.max(1,rows.length-1),rows[0].length).format.rowHeight =
      supplement==='S4' ? 56 : (supplement==='S2' || supplement==='S3' ? 42 : 30);
    for (let j=0;j<rows[0].length;j++) {
      let max = rows[0][j].length;
      for (const row of rows.slice(1)) if (typeof row[j]==='string') max=Math.max(max, Math.min(70,row[j].length));
      const narrative = /note|interpretation|limitation|summary|title|coverage|category/i.test(rows[0][j]);
      const width=narrative ? Math.min(52,Math.max(24,max+2)) : Math.min(27,Math.max(14,max+2));
      ws.getRangeByIndexes(0,j,rows.length,1).format.columnWidth=width;
      if (/mean|difference|CI|rate|Pearson/i.test(rows[0][j]))
        ws.getRangeByIndexes(1,j,Math.max(1,rows.length-1),1).setNumberFormat('0.000');
      if (/P value|sign-flip P|BH q/i.test(rows[0][j]))
        ws.getRangeByIndexes(1,j,Math.max(1,rows.length-1),1).setNumberFormat('0.0000');
    }
    // Data columns remain numeric and sortable in the delivered workbooks.
    for (let i=1;i<rows.length;i++) {
      if (i%2===0) ws.getRangeByIndexes(i,0,1,rows[0].length).format.fill='#F2F6FA';
    }
    r.format.borders={preset:'inside',style:'thin',color:'#E4E8ED'};
  }
  wb.recalculate();
  const firstName=Object.keys(sheets)[0];
  const check=await wb.inspect({kind:'table',range:`'${firstName}'!A1:D5`,include:'values',
                                tableMaxRows:5,tableMaxCols:4,maxChars:900});
  console.log(supplement,check.ndjson.slice(0,900));
  for (const sheetName of Object.keys(sheets)) {
    try {
      const preview=await wb.render({sheetName,range:'A1:F8',scale:1.2,format:'png'});
      await fs.writeFile(path.join(outDir,`${supplement}_${sheetName.replaceAll(' ','_')}_preview.png`),
                         new Uint8Array(await preview.arrayBuffer()));
    } catch (err) { console.log('Preview unavailable for',supplement,sheetName,String(err).slice(0,180)); }
  }
  const xlsx=await SpreadsheetFile.exportXlsx(wb);
  await xlsx.save(path.join(outDir,`${supplement}_Table_revised.xlsx`));
}
console.log(outDir);
