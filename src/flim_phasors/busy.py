"""Background work with cancel support for the Qt GUI.

Long-running FLIM loads and phasor computations run off the main thread so the
UI stays responsive. Workers poll a :class:`CancelToken` to exit early when the
user dismisses the progress dialog.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class CancelledError(Exception):
    """Raised when the user cancels a long-running job.

    Signals cooperative cancellation of background work started via
    :func:`run_busy_qt`: either the worker itself raises this after noticing
    :attr:`CancelToken.cancelled`, or ``run_busy_qt`` raises it on the caller's
    behalf when the dialog is dismissed before the job finishes. Callers
    typically catch this to suppress error dialogs for user-initiated aborts.
    """


class CancelToken:
    """Thread-safe cancellation flag checked by worker loops.

    A single instance is shared between the GUI thread (which calls
    :meth:`cancel` when the user dismisses the progress dialog) and the
    background worker thread (which polls :attr:`cancelled` or calls
    :meth:`check` between processing steps to exit early).
    """

    def __init__(self):
        """Create a token that starts in the non-cancelled state.

        Backed by a :class:`threading.Event` so ``cancel()`` from the GUI
        thread is immediately visible to a worker polling on another thread
        without extra locking.
        """
        self._event = threading.Event()

    def cancel(self):
        """Signal cooperative cancellation to any worker polling this token.

        Does not forcibly stop the worker thread; it only sets the flag that
        :attr:`cancelled` and :meth:`check` observe, so the worker must reach
        a check point to actually abort.
        """
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called.

        Backed by :meth:`threading.Event.is_set`, so this is safe to poll
        from a worker thread without additional locking while the GUI
        thread calls :meth:`cancel` concurrently. Worker loops typically
        check this (or call :meth:`check`) between chunks of long-running
        work rather than only once at the start.

        Returns:
            True once cancellation has been requested; stays True for the
            lifetime of this token (there is no un-cancel).
        """
        return self._event.is_set()

    def check(self):
        """Raise :class:`CancelledError` if cancellation was requested.

        Convenience wrapper over :attr:`cancelled` for worker code that
        wants to abort immediately rather than branch on a boolean; calling
        this at natural checkpoints (e.g. between processing steps of a long
        FLIM load) lets :func:`run_busy_qt` unwind cleanly via the normal
        exception path instead of needing a separate polling loop.

        Raises:
            CancelledError: When :attr:`cancelled` is true.
        """
        if self.cancelled:
            raise CancelledError("Cancelled by user")


# --- unused: uncomment if needed ---
# def run_in_thread(
#     fn: Callable[[], T],
#     *,
#     cancel: CancelToken | None = None,
#     on_progress: Callable[[], None] | None = None,
# ) -> T:
#     """Run fn in a worker thread; optional cancel checks between progress callbacks."""
#     result: list[T] = []
#     error: list[BaseException] = []
#
#     def target():
#         try:
#             if cancel is not None:
#                 old = getattr(fn, "__cancel__", None)
#                 if old is None and hasattr(fn, "__self__"):
#                     pass
#             result.append(fn())
#         except BaseException as exc:
#             error.append(exc)
#
#     thread = threading.Thread(target=target, daemon=True)
#     thread.start()
#     while thread.is_alive():
#         if cancel is not None and cancel.cancelled:
#             pass
#         if on_progress:
#             on_progress()
#         thread.join(timeout=0.05)
#     if error:
#         raise error[0]
#     return result[0]


def run_busy_qt(
    parent,
    message: str,
    fn: Callable[[], T],
    *,
    log_fn: Callable[[str], None] | None = None,
    cancellable: bool = True,
    progress_hook: Callable[[], None] | None = None,
    cancel_out: list[CancelToken] | None = None,
) -> tuple[T, float]:
    """Run *fn* off the GUI thread behind a modal progress dialog.

    Starts ``fn`` in a daemon :class:`threading.Thread` while the GUI thread
    keeps pumping ``QApplication.processEvents`` so the window stays
    responsive (and repaints the progress dialog) while the background work
    runs; a :class:`CancelToken` is created per call and wired to the
    dialog's Cancel button so dismissing it does not forcibly kill the
    thread but instead lets cooperative checkpoints inside ``fn`` notice and
    raise :class:`CancelledError`.

    Args:
        parent: Qt parent widget for the progress dialog.
        message: Status text shown while *fn* runs.
        fn: Callable executed in a background thread (e.g. load FLIM file).
        log_fn: Optional callback invoked with *message* before work starts.
        cancellable: When false, hide the Cancel button.
        progress_hook: Optional callback polled on the GUI thread while waiting.
        cancel_out: If given, cleared and filled with the :class:`CancelToken`
            so callers can wire additional cancel paths.

    Returns:
        A ``(result, elapsed_seconds)`` tuple where *result* is the return
        value of *fn*.

    Raises:
        CancelledError: If the user cancels before *fn* completes.
        Exception: Any exception raised by *fn* is re-raised after the dialog
            closes.
    """
    from PySide6 import QtCore, QtWidgets

    if log_fn:
        log_fn(message)
    cancel = CancelToken()
    if cancel_out is not None:
        cancel_out.clear()
        cancel_out.append(cancel)
    dlg = QtWidgets.QProgressDialog(message, "Cancel", 0, 0, parent)
    dlg.setWindowTitle("Loading file")
    dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    if not cancellable:
        dlg.setCancelButton(None)
    else:
        dlg.canceled.connect(cancel.cancel)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    t0 = time.perf_counter()
    result: list[T] = []
    error: list[BaseException] = []
    completed = False

    def target():
        """Worker thread entry: run ``fn`` and capture result or exception.

        Runs on the background :class:`threading.Thread` started by
        ``run_busy_qt``; any return value or raised exception is stashed in
        the enclosing ``result``/``error`` lists (rather than returned or
        re-raised directly) so the GUI thread can retrieve it after joining.
        """
        nonlocal completed
        try:
            result.append(fn())
            completed = True
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if progress_hook:
            progress_hook()
        if cancel.cancelled:
            dlg.setLabelText(f"{message} (cancelling…)")
        thread.join(timeout=0.05)
    thread.join()
    # Closing QProgressDialog emits canceled on some platforms — do not treat that as user cancel.
    if cancellable:
        try:
            dlg.canceled.disconnect(cancel.cancel)
        except (TypeError, RuntimeError):
            pass
    dlg.blockSignals(True)
    dlg.close()
    dlg.blockSignals(False)
    if cancel.cancelled and not completed and not error:
        raise CancelledError("Cancelled by user")
    if error:
        raise error[0]
    if not result:
        raise CancelledError("Cancelled by user")
    return result[0], time.perf_counter() - t0
