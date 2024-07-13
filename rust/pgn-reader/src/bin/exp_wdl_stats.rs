// use buffered_reader::BufferedReader;
// use std::collections::HashMap;
// use std::fmt::Debug;
// use std::fs::File;
// use std::io::Write;
// use std::time::Instant;
//
// use zstd::Decoder;
//
// use pgn_reader::PgnReader;
//
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // let path = r#"\\192.168.0.10\Documents\Download\lichess_db_standard_rated_2023-09.pgn.zst"#;
//     let path = r#"D:\Documents\A0\lichess_db_standard_rated_2023-09.pgn.zst"#;
//     let total_games = 93_218_629;
//
//     let decoder = Decoder::new(File::open(path)?)?;
//     let reader = PgnReader::new_generic(decoder);
//
//     basic(reader, total_games);
//     // all(reader, total_games);
//
//     Ok(())
// }
//
// #[derive(Default, Clone)]
// struct Bucket {
//     tc: HashMap<String, u64>,
//     result: HashMap<String, u64>,
// }
//
// fn all<R: BufferedReader<()>>(mut reader: PgnReader<R>, total_games: u64) {
//     let mut elo_buckets = vec![Bucket::default(); 100];
//     let mut total_seen = 0;
//     let mut start = Instant::now();
//
//     while let Some(game) = reader.next_game().unwrap() {
//         total_seen += 1;
//
//         let white_elo = game.header("WhiteElo").unwrap().parse::<u32>().unwrap();
//         let black_elo = game.header("BlackElo").unwrap().parse::<u32>().unwrap();
//         let time_control = game.header("TimeControl").unwrap();
//         let result = game.header("Result").unwrap();
//
//         // require same bucket
//         let bucket = white_elo / 100;
//         if black_elo / 100 != bucket {
//             continue;
//         }
//
//         let bucket = &mut elo_buckets[bucket as usize];
//         map_increment(&mut bucket.tc, time_control);
//         map_increment(&mut bucket.result, result);
//
//         // if total_seen > 1_000_000 {
//         //     break;
//         // }
//
//         if total_seen % 100_000 == 0 {
//             let eta = start.elapsed() * (total_games - total_seen) as u32 / total_seen as u32;
//
//             println!(
//                 "count: {}, games/s: {:.2}, expected: {:.4}, eta: {:?}",
//                 total_seen,
//                 total_seen as f64 / start.elapsed().as_secs_f64(),
//                 total_games as f64 * total_seen as f64 / total_seen as f64,
//                 eta,
//             );
//         }
//     }
//
//     let elos: Vec<_> = (0..elo_buckets.len()).map(|i| i * 100 + 50).collect();
//
//     let mut tcs: Vec<_> = elo_buckets.iter().flat_map(|b| b.tc.keys()).collect();
//     tcs.sort();
//     tcs.dedup();
//     tcs.retain(|tc| elo_buckets.iter().map(|b| b.tc.get(*tc).unwrap_or(&0)).sum::<u64>() > 10);
//
//     let results = vec!["-", "1-0", "1/2-1/2", "0-1"];
//
//     let tc_counts: Vec<Vec<_>> = elo_buckets
//         .iter()
//         .map(|b| tcs.iter().map(|tc| b.tc.get(*tc).unwrap_or(&0)).collect())
//         .collect();
//     let result_counts: Vec<Vec<_>> = elo_buckets
//         .iter()
//         .map(|b| {
//             results
//                 .iter()
//                 .map(|result| b.result.get(*result).unwrap_or(&0))
//                 .collect()
//         })
//         .collect();
//
//     let mut f = File::create("output.txt").unwrap();
//
//     writeln!(&mut f, "elos = {:?}", elos).unwrap();
//     writeln!(&mut f, "results = {:?}", results).unwrap();
//
//     writeln!(&mut f, "tcs = {:?}", tcs).unwrap();
//     writeln!(&mut f, "tc_counts = {:?}", tc_counts).unwrap();
//     writeln!(&mut f, "result_counts = {:?}", result_counts).unwrap();
//
//     f.flush().unwrap()
// }
//
// fn basic<R: BufferedReader<()>>(mut reader: PgnReader<R>, total_games: u64) {
//     let mut count: u64 = 0;
//     let mut accepted: u64 = 0;
//     let mut start = Instant::now();
//
//     // let mut tc_counts = HashMap::new();
//     // let mut result_counts = HashMap::new();
//
//     let mut result_60 = HashMap::new();
//     let mut result_180 = HashMap::new();
//
//     while let Some(game) = reader.next_game().unwrap() {
//         let white_elo = game.header("WhiteElo").unwrap().parse::<u32>().ok().unwrap();
//         let black_elo = game.header("BlackElo").unwrap().parse::<u32>().ok().unwrap();
//         let time_control = game.header("TimeControl").unwrap();
//         // let termination = game.header("Termination").unwrap();
//         let result = game.header("Result").unwrap();
//
//         let elo_range = 2500..2600;
//         let mut accept = false;
//         if elo_range.contains(&white_elo) && elo_range.contains(&black_elo) {
//             //     println!("Accepted game");
//             //     println!("{:?}", game);
//             // map_increment(&mut tc_counts, game.header("TimeControl").unwrap());
//             // map_increment(&mut result_counts, game.header("Result").unwrap());
//
//             if time_control == "60+0" {
//                 map_increment(&mut result_60, result);
//                 accept = true;
//             } else if time_control == "180+0" {
//                 map_increment(&mut result_180, result);
//                 accept = true;
//             }
//         }
//
//         // count throughput
//         count += 1;
//         accepted += accept as u64;
//
//         if count % 1_000_000 == 0 {
//             let eta = start.elapsed() * (total_games - count) as u32 / count as u32;
//
//             println!(
//                 "count: {}, games/s: {:.2}, accepted: {:.4} {}, expected: {:.4}, eta: {:?}",
//                 count,
//                 count as f64 / start.elapsed().as_secs_f64(),
//                 accepted,
//                 accepted as f64 / count as f64,
//                 total_games as f64 * accepted as f64 / count as f64,
//                 eta,
//             );
//
//             // println!("TC distribution:");
//             // map_print(&tc_counts);
//             // println!("Result distribution:");
//             // map_print(&result_counts);
//
//             println!("Result 60+0:");
//             map_print(&result_60);
//             println!("Result 180+0:");
//             map_print(&result_180);
//         }
//     }
// }
//
// fn map_increment(map: &mut HashMap<String, u64>, key: &str) {
//     match map.get_mut(key) {
//         Some(count) => *count += 1,
//         None => {
//             map.insert(key.to_string(), 1);
//         }
//     }
// }
//
// fn map_print(map: &HashMap<String, u64>) {
//     let mut keys: Vec<_> = map.keys().collect();
//     let total = map.values().sum::<u64>();
//
//     keys.sort();
//     for tc in keys {
//         println!("  {}: {} {:.2}%", tc, map[tc], 100.0 * map[tc] as f64 / total as f64);
//     }
// }

fn main() {}
