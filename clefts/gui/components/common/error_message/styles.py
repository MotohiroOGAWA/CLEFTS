ERROR_MESSAGE_CSS = """
.clefts-generic-error-toast {
    position: fixed;

    top: 18px;
    right: 18px;

    z-index: 9999;

    background: #fff3f3;
    border-left: 4px solid #c0392b;

    color: #8a1f16;

    padding: 12px 14px;

    border-radius: 6px;

    box-shadow: 0 8px 24px rgba(0,0,0,.12);

    min-width: 240px;
    max-width: 420px;

    font-size: 14px;
    line-height: 1.4;

    animation:
        clefts-error-toast-in .16s ease-out,
        clefts-error-toast-out .18s ease-in
            var(--clefts-error-duration, 3s)
            forwards;
}

@keyframes clefts-error-toast-in {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes clefts-error-toast-out {
    to {
        opacity: 0;
        transform: translateY(-8px);
        visibility: hidden;
    }
}
"""
