* Fixed ``scripts/update_release_notes.sh`` to produce a single commit instead of two,
  and to correctly update ``docs/release_notes/index.rst`` (previously wrote to a
  nonexistent ``docs/release_notes.rst`` file).

* Fixed ``scripts/combine_release_notes.sh`` to use the ``v`` prefix in release note
  titles for consistency, and to concatenate full fragment content instead of only
  extracting bullet-point lines.
