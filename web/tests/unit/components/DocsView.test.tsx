import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DocsView } from '../../../src/components/DocsView';
import type { DocIndex } from 'go-ui';

// A minimal DocIndex the stubbed fetch returns for DocsApp's doc.json request.
const DOC_INDEX: DocIndex = {
  module: 'github.com/malcolmston/pandas',
  packages: [
    {
      importPath: 'github.com/malcolmston/pandas',
      name: 'pandas',
      synopsis: 'Package pandas provides pandas-style data structures for Go.',
      doc: 'Package pandas provides pandas-style data structures for Go.',
      consts: [],
      vars: [],
      types: [
        {
          name: 'DataFrame',
          signature: 'type DataFrame struct{}',
          doc: 'DataFrame is an ordered set of equal-length named columns.',
          consts: [],
          vars: [],
          funcs: [],
          methods: [],
        },
      ],
      funcs: [{ name: 'FromMap', signature: 'func FromMap(data map[string][]any, order []string) (*DataFrame, error)', doc: 'FromMap builds a DataFrame from column data.' }],
    },
  ],
};

describe('DocsView', () => {
  beforeEach(() => {
    // DocsApp fetches doc.json; return the small index.
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      if (String(input).includes('doc.json')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(DOC_INDEX) } as Response);
      }
      return new Promise<Response>(() => {});
    }) as unknown as typeof fetch;
  });

  it('renders the inline React API reference from the fetched doc.json', async () => {
    const { container } = render(<DocsView />);
    expect(container.querySelector('#view-docs')).not.toBeNull();
    expect(
      screen.getByRole('heading', { level: 2, name: /API documentation/ }),
    ).toBeInTheDocument();

    // DocsApp fetches asynchronously, then renders the package view + symbols.
    expect(await screen.findByRole('heading', { name: /package pandas/ })).toBeInTheDocument();
    expect(container.querySelector('#sym-FromMap'), 'func FromMap symbol card').not.toBeNull();
    expect(container.querySelector('#sym-DataFrame'), 'type DataFrame symbol card').not.toBeNull();

    // The secondary link to the raw generated static HTML remains.
    expect(screen.getByRole('link', { name: /Open the raw generated HTML/ })).toHaveAttribute('href', './api/');
  });
});
