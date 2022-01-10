use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

/// A newtype that implements debug by only printing the name of the contained type.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WrapDebug<T>(pub T);

impl<T> WrapDebug<T> {
    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T> Debug for WrapDebug<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", std::any::type_name::<T>())
    }
}

impl<T> From<T> for WrapDebug<T> {
    fn from(value: T) -> Self {
        WrapDebug(value)
    }
}

impl<T> Deref for WrapDebug<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for WrapDebug<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A newtype that forwards the debug call but clears the formatting flags.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClearDebug<T>(pub T);

impl<T> ClearDebug<T> {
    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T: Debug> Debug for ClearDebug<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner())
    }
}

impl<T> From<T> for ClearDebug<T> {
    fn from(value: T) -> Self {
        ClearDebug(value)
    }
}

impl<T> Deref for ClearDebug<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for ClearDebug<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
