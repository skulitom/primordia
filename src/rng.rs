//! Tiny deterministic RNG (SplitMix64) for seeding simulations on the CPU.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self(seed ^ 0x5DEE_CE66_D1CE_4E5B);
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A seed for a new run, below 2^53 so that it survives tools that read
    /// JSON numbers as doubles (JavaScript, jq) unchanged.
    pub fn next_seed(&mut self) -> u64 {
        self.next_u64() >> (64 - SEED_BITS)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Standard normal sample (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u32) as usize]
    }
}

/// Bits of the seeds Primordia draws itself ([`Rng::next_seed`], [`time_seed`]):
/// every integer below 2^53 is exact as a double, so recipes stay reproducible
/// after a round trip through JavaScript or jq. Seeds typed by hand may be any `u64`.
pub const SEED_BITS: u32 = 53;

/// A seed derived from the wall clock, for "surprise me" resets (below 2^53, see [`SEED_BITS`]).
pub fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678)
        & ((1 << SEED_BITS) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawn_seeds_are_exact_as_doubles() {
        let limit = 1u64 << SEED_BITS;
        let mut rng = Rng::new(7);
        let seeds: Vec<u64> = (0..1000).map(|_| rng.next_seed()).collect();
        assert!(seeds.iter().all(|&s| s < limit && s as f64 as u64 == s));
        assert!(seeds.iter().any(|&s| s >= limit / 2), "the whole 53-bit range is used");
        assert!(time_seed() < limit);
        assert_ne!(Rng::new(7).next_seed(), Rng::new(8).next_seed());
    }
}
