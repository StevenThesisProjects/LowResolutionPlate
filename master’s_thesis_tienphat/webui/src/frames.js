// Image previews for the selected input, in the order the API reads the frames:
// natural sort by file name; from a track zip only the lr-* frames (HR frames are skipped).
import JSZip from 'jszip';

const IMAGE = /\.(jpe?g|png)$/i;
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
const basename = (path) => path.split('/').pop();

export const isZip = (file) => /\.zip$/i.test(file.name);

export async function framePreviews(files) {
  if (files.length === 1 && isZip(files[0])) {
    const zip = await JSZip.loadAsync(files[0]);
    let names = Object.keys(zip.files).filter((n) => {
      const base = basename(n);
      return !zip.files[n].dir && IMAGE.test(n) && !n.includes('__MACOSX') && !base.startsWith('.');
    });
    const lr = names.filter((n) => basename(n).toLowerCase().startsWith('lr-'));
    names = (lr.length ? lr : names).sort(natural);
    return Promise.all(names.map(async (n) => ({
      name: basename(n),
      url: URL.createObjectURL(await zip.files[n].async('blob')),
    })));
  }
  return [...files]
    .filter((f) => IMAGE.test(f.name) || f.type.startsWith('image/'))
    .sort((a, b) => natural(a.name, b.name))
    .map((f) => ({ name: f.name, url: URL.createObjectURL(f) }));
}

export function revokePreviews(previews) {
  previews.forEach((p) => URL.revokeObjectURL(p.url));
}
