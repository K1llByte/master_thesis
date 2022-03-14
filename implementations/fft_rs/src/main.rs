use std::f32;
use num::Complex;

// fn fft<const N: usize>(p: &[f32; N]) -> [f32; N] {
//     let n = p.len();
//     if n == 1 {
//         return p.clone()
//     }
//     let omega = num::Complex::new(0., 2. * std::f32::consts::PI / n as f32);
//     // let omega = f32::consts::E.powf(2*pi*i);
//     let mut pe: [f32] = vec![0f32; N/2].try_into().unwrap();
//     let mut po: [f32] = vec![0f32; N/2].try_into().unwrap();
//     for (i, &c) in p.iter().enumerate() {
//         if i % 2 == 0 {
//             pe[i/2] = c;
//         }
//         else {
//             po[(i-1)/2] = c;
//         }
//     }
//     println!("pe: {:?}", pe);
//     println!("po: {:?}", po);

//     let (ye, to) = (fft(pe), fft(&po));

//     let mut y = vec![0f32; N];
//     for (j, (ye, yo)) in pe.iter().zip(po).enumerate() {
//         y[j] = ye;
//     }

//     return p.clone()
// }

fn fft<const N: usize>(p: Vec<f32>) -> Vec<f32> {
    if N == 1 {
        *p
    } else {
        let omega = Complex::new(0., 2. * std::f32::consts::PI / p.len() as f32);
        let mut buf = vec![0f32; N];
        let (pe, po) = buf.split_at_mut(N / 2);
        for ((even_odd, e), o) in p.chunks(2).zip(pe.iter_mut()).zip(po.iter_mut()) {
            *e = even_odd[0];
            *o = even_odd[1];
        }
        println!("pe: {:?}", pe);
        println!("po: {:?}", po);
        // let (ye, yo) = (fft(&pe), fft(po));
        
        *p
    }
}

fn main() {
    // let arr: [f32; 4] = [0., 1., 2., 3.];
    // let res = fft(&arr);
    // println!("res: {:?}", res);

    let omega = num::Complex::new(0., 2. * std::f32::consts::PI / 4 as f32);
    println!("{}", omega);
}