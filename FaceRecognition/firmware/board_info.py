"""Read-only diagnostics for Arduino Nicla Vision / OpenMV 5.0.1.

Run this file on the board, using the OpenMV host tool's --script option.
It does not initialize the camera, modify ROMFS, or write any board files.
"""

import gc
import json
import os
import sys
import time
import omv

EXPECTED_FIRMWARE = "5.0.1"
MAX_ROM_ENTRIES = 32
MAX_FLASH_ENTRIES = 32


def emit(event, fields):
    fields["schema_version"] = 1
    fields["event"] = event
    fields["ticks_ms"] = time.ticks_ms()
    print(json.dumps(fields))


def report_optional_error(component, error):
    emit("diagnostic_unavailable", {
        "component": component,
        "error": str(error)[:160],
    })


def report_flash_storage():
    """Report writable flash capacity and filenames without reading contents."""
    try:
        stats = os.statvfs("/flash")
        block_size = stats[1] or stats[0]
        emit("flash_storage", {
            "filesystem_block_size_bytes": block_size,
            "total_bytes": block_size * stats[2],
            "free_bytes": block_size * stats[3],
            "available_bytes": block_size * stats[4],
        })
    except Exception as error:
        report_optional_error("flash_statvfs", error)

    try:
        entry_count = 0
        truncated = False
        for name in os.listdir("/flash"):
            if entry_count >= MAX_FLASH_ENTRIES:
                truncated = True
                break
            try:
                size_bytes = os.stat("/flash/" + name)[6]
            except Exception:
                size_bytes = None
            emit("flash_entry", {
                "name": name[:96],
                "size_bytes": size_bytes,
            })
            entry_count += 1
        emit("flash_listing_complete", {
            "entries_reported": entry_count,
            "truncated": truncated,
        })
    except Exception as error:
        report_optional_error("flash_listing", error)


def main():
    gc.collect()
    version = omv.version_string()
    emit("board_info", {
        "board_type": omv.board_type(),
        "architecture": omv.arch(),
        "board_id": omv.board_id(),
        "openmv_version": version,
        "expected_openmv_version": EXPECTED_FIRMWARE,
        "micropython_version": sys.version,
        "gc_free_bytes": gc.mem_free(),
        "gc_allocated_bytes": gc.mem_alloc(),
    })
    if version != EXPECTED_FIRMWARE:
        emit("version_warning", {
            "expected": EXPECTED_FIRMWARE,
            "actual": version,
        })

    try:
        import cpufreq
        emit("clock_info", {
            "current_frequencies_mhz": cpufreq.get_current_frequencies(),
            "supported_cpu_frequencies_mhz": cpufreq.get_supported_frequencies(),
        })
    except Exception as error:
        report_optional_error("cpufreq", error)

    try:
        import umalloc
        # OpenMV 5.0.1 returns seven fields; this pool is separate from GC.
        stats = umalloc.stats(0)
        emit("uma_pool", {
            "pool": 0,
            "used_blocks": stats[0],
            "free_blocks": stats[1],
            "persistent_blocks": stats[2],
            "used_bytes": stats[3],
            "free_bytes": stats[4],
            "persistent_bytes": stats[5],
            "peak_bytes": stats[6],
        })
    except Exception as error:
        report_optional_error("umalloc", error)

    try:
        import vfs
        count = vfs.rom_ioctl(1)
        emit("romfs_info", {"partition_count": count})
        for partition in range(min(count, 4)):
            emit("romfs_partition", {
                "partition": partition,
                "size_bytes": len(vfs.rom_ioctl(2, partition)),
            })
    except Exception as error:
        report_optional_error("romfs_partition_query", error)

    report_flash_storage()

    try:
        entry_count = 0
        truncated = False
        for entry in os.ilistdir("/rom"):
            if entry_count >= MAX_ROM_ENTRIES:
                truncated = True
                break
            name = entry[0]
            emit("romfs_entry", {
                "name": name[:96],
                "type": entry[1],
                "size_bytes": os.stat("/rom/" + name)[6],
            })
            entry_count += 1
        emit("romfs_listing_complete", {
            "entries_reported": entry_count,
            "truncated": truncated,
        })
    except Exception as error:
        report_optional_error("romfs_listing", error)

    emit("diagnostics_complete", {
        "gc_free_bytes": gc.mem_free(),
        "gc_allocated_bytes": gc.mem_alloc(),
    })


main()
