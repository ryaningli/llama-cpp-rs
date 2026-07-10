//! Safe wrapper around `llama_batch`.

use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};
use std::marker::PhantomData;

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch<'a> {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    /// The logits that are initialized. Used by [`LlamaContext`] to ensure that only initialized logits are accessed.
    pub(crate) initialized_logits: Vec<i32>,
    #[allow(clippy::doc_markdown)]
    /// The llama_cpp batch. always initialize by `llama_cpp_sys_2::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
    phantom: PhantomData<&'a [LlamaToken]>,
}

/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    /// Empty buffer is provided for [`LlamaBatch::get_one`]
    #[error("Empty buffer")]
    EmptyBuffer,
}

impl<'a> LlamaBatch<'a> {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// add a token to the batch for sequences `seq_ids` at position `pos`. If `logits` is true, the
    /// token will be initialized and can be read from after the next decode.
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a usize
    /// - [`seq_ids.len()`] does not fit into a [`llama_seq_id`]
    ///
    /// # Errors
    ///
    /// returns a error if there is insufficient space in the buffer
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token   [batch.n_tokens] = id;
            self.llama_batch.token.add(offset_usize).write(id);
            // batch.pos     [batch.n_tokens] = pos,
            self.llama_batch.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            // batch.logits  [batch.n_tokens] = logits;
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        // batch.n_tokens++;
        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    /// Add a sequence of tokens to the batch for the given sequence id. If `logits_all` is true, the
    /// tokens will be initialized and can be read from after the next decode.
    ///
    /// Either way the last token in the sequence will have its logits set to `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a [`usize`]
    /// - [`n_tokens - 1`] does not fit into a [`llama_pos`]
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let n_tokens_0 =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        let n_tokens = tokens.len();

        if self.allocated < n_tokens_0 + n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let last_index = llama_pos::try_from(n_tokens.saturating_sub(1))
            .expect("cannot fit n_tokens into a llama_pos");
        for (i, token) in (0..).zip(tokens.iter()) {
            self.add(*token, i, &[seq_id], logits_all || i == last_index)?;
        }

        Ok(())
    }

    /// Create a new `LlamaBatch` that can contain up to `n_tokens` tokens.
    ///
    /// # Arguments
    ///
    /// - `n_tokens`: the maximum number of tokens that can be added to the batch
    /// - `n_seq_max`: the maximum number of sequences that can be added to the batch (generally 1 unless you know what you are doing)
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is greater than `i32::MAX`.
    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        Self::new_with_embd(n_tokens, 0, n_seq_max)
    }

    /// Create a new `LlamaBatch` with embedding dimension, for use with [`Self::set_embd`].
    ///
    /// When `embd_dim > 0`, the batch will allocate an embedding buffer suitable for
    /// embedding-based inference (e.g., audio/ASR models).
    ///
    /// # Arguments
    ///
    /// - `n_tokens`: the maximum number of tokens that can be added to the batch
    /// - `embd_dim`: the embedding dimension per token (0 = token-based batch, no embd buffer)
    /// - `n_seq_max`: the maximum number of sequences that can be added to the batch
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` or `embd_dim` is greater than `i32::MAX`.
    #[must_use]
    pub fn new_with_embd(n_tokens: usize, embd_dim: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let embd_dim_i32 = i32::try_from(embd_dim).expect("cannot fit embd_dim into a i32");
        let batch = unsafe { llama_batch_init(n_tokens_i32, embd_dim_i32, n_seq_max) };

        LlamaBatch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// ``llama_batch_get_one``
    /// Return batch for single sequence of tokens
    ///
    /// NOTE: this is a helper function to facilitate transition to the new batch API
    ///
    /// # Errors
    /// If the provided token buffer is empty.
    ///
    /// # Panics
    /// If the number of tokens in ``tokens`` exceeds [`i32::MAX`].
    pub fn get_one(tokens: &'a [LlamaToken]) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }
        let batch = unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            llama_cpp_sys_2::llama_batch_get_one(
                ptr,
                tokens
                    .len()
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX"),
            )
        };
        let batch = Self {
            allocated: 0,
            initialized_logits: vec![(tokens.len() - 1)
                .try_into()
                .expect("number of tokens exceeds i32::MAX + 1")],
            llama_batch: batch,
            phantom: PhantomData,
        };
        Ok(batch)
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }

    /// Batch-set embedding data into the batch, for audio/embedding input scenarios (e.g., Qwen3 ASR).
    ///
    /// Directly writes to the underlying `llama_batch` `embd`, `pos`, `n_seq_id`, `seq_id`,
    /// and `logits` fields. After this call, `n_tokens` is set to `embd_data.len() / dim`
    /// (or `positions.len() / stride` if stride is provided).
    ///
    /// # Arguments
    ///
    /// * `embd_data` - Flattened embedding vectors (length = n_tokens * dim)
    /// * `dim` - Embedding dimension per token
    /// * `positions` - Position encoding array for each token
    /// * `stride` - Optional position stride (e.g., Qwen3 uses 4x stride)
    /// * `seq_ids` - Sequence IDs for each token. Two modes:
    ///   - `&[id]` (length 1): all tokens share the same seq_id
    ///   - `&[id0, id1, ...]` (length == actual_n_tokens): each token gets its own seq_id
    pub fn set_embd(
        &mut self,
        embd_data: &[f32],
        dim: usize,
        positions: &[llama_pos],
        stride: Option<i32>,
        seq_ids: &[i32],
    ) -> Result<(), BatchAddError> {
        let n_tokens = embd_data.len() / dim;

        let actual_n_tokens = match stride {
            Some(s) => positions.len() / (s as usize),
            None => n_tokens,
        };

        if actual_n_tokens > self.allocated {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        // Validate seq_ids length: must be 1 (all tokens share) or actual_n_tokens (per-token)
        if seq_ids.len() != 1 && seq_ids.len() != actual_n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let embd_ptr = self.llama_batch.embd;
        if embd_ptr.is_null() {
            return Err(BatchAddError::EmptyBuffer);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(embd_data.as_ptr(), embd_ptr, embd_data.len());
        }

        let pos_ptr = self.llama_batch.pos;
        if !pos_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(positions.as_ptr(), pos_ptr, positions.len());
            }
        }

        self.llama_batch.n_tokens = actual_n_tokens as i32;

        let n_seq_id_ptr = self.llama_batch.n_seq_id;
        let seq_id_ptr = self.llama_batch.seq_id;
        let logits_ptr = self.llama_batch.logits;

        if !n_seq_id_ptr.is_null() && !seq_id_ptr.is_null() && !logits_ptr.is_null() {
            for i in 0..actual_n_tokens {
                let sid = if seq_ids.len() == 1 {
                    seq_ids[0]
                } else {
                    seq_ids[i]
                };
                unsafe {
                    *n_seq_id_ptr.add(i) = 1;
                    let seq_id_arr = *seq_id_ptr.add(i);
                    if !seq_id_arr.is_null() {
                        *seq_id_arr = sid;
                    }
                    *logits_ptr.add(i) = if i == actual_n_tokens - 1 { 1 } else { 0 };
                }
            }
        }

        // Track the last token's logits as initialized (matches add() behavior)
        self.initialized_logits.clear();
        if actual_n_tokens > 0 {
            self.initialized_logits.push(actual_n_tokens as i32 - 1);
        }

        Ok(())
    }

    /// Set the logits flag at a specific token position.
    ///
    /// Used in multi-sequence batch scenarios to enable logits at additional positions
    /// after calling `set_embd()`. Since `set_embd()` only sets logits for the last
    /// token, this method is needed to enable logits for the last token of each
    /// individual sequence when multiple sequences are merged into one batch.
    ///
    /// # Arguments
    /// * `idx` - Token index in the batch (0-based)
    /// * `logits` - Whether to enable logits computation
    ///
    /// # Panics
    /// Panics if `idx` is negative.
    pub fn set_logits_at(&mut self, idx: i32, logits: bool) {
        let idx_usize = usize::try_from(idx).expect("idx must be non-negative");
        if idx_usize >= self.allocated {
            return;
        }
        unsafe {
            self.llama_batch.logits.add(idx_usize).write(i8::from(logits));
        }
        if logits {
            if !self.initialized_logits.contains(&idx) {
                self.initialized_logits.push(idx);
            }
        } else {
            self.initialized_logits.retain(|l| l != &idx);
        }
    }
}

impl<'a> Drop for LlamaBatch<'a> {
    /// Drops the `LlamaBatch`.
    ///
    /// ```
    /// # use llama_cpp_2::llama_batch::LlamaBatch;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let batch = LlamaBatch::new(512, 1);
    /// // frees the memory associated with the batch. (allocated by llama.cpp)
    /// drop(batch);
    /// # Ok(())
    /// # }
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_batch_free(self.llama_batch);
            }
        }
    }
}
