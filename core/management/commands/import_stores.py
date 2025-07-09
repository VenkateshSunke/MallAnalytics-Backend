import pandas as pd
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import MallStore


class Command(BaseCommand):
    help = 'Import stores from Excel file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='./Stores.xlsx',
            help='Path to the Excel file (default: ./Stores.xlsx)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing stores before importing'
        )

    def handle(self, *args, **options):
        # Category mapping from Spanish to English
        category_mapping = {
            'Artículos Personales': 'Personal Items',
            'Moda': 'Fashion',
            'Calzado': 'Footwear',
            'Cuidado Personal': 'Personal Care',
            'Deportes': 'Sports',
            'Comidas y Bebidas': 'Food and Beverages',
            'Arte, Cultura y Educación': 'Art, Culture and Education',
            'Tecnología': 'Technology',
            'Entretenimiento': 'Entertainment',
            'Hogar': 'Home',
            'Niños': 'Kids',
            'Servicios': 'Services',
            'Tiendas Especializadas': 'Specialized Stores',
            'Tiendas por Departamentos': 'Department Stores',
            'Supermercado': 'Supermarket',
            'Salud': 'Health',
            'Viajes': 'Travel',
            'Vehículos': 'Vehicles',
            'moda': 'Fashion',  # lowercase variant
        }

        # Subcategory mapping from Spanish to English
        subcategory_mapping = {
            'Artículos de cuero': 'Leather goods',
            'Ópticas': 'Optics',
            'Moda unisex': 'Unisex fashion',
            'Calzado unisex': 'Unisex footwear',
            'Perfumería, belleza y maquillaje': 'Perfumery, beauty and makeup',
            'Calzado femenino': "Women's footwear",
            'Artículos y moda deportiva': 'Sports goods and fashion',
            'Heladería': 'Ice cream shops',
            'Moda femenina': "Women's fashion",
            'Joyería y relojería': 'Jewelry and watches',
            'Cafeterías': 'Cafeterias',
            'Electrónicos': 'Electronics',
            'Bisutería': 'Costume jewelry',
            'Accesorios para niños': "Kids' accessories",
            'Postres y dulces': 'Desserts and sweets',
            'Librerías y útiles de escritorio': 'Bookstores and stationery',
            'Videojuegos y música': 'Video games and music',
            'Moda masculina': "Men's fashion",
            'Maletas, carteras y mochilas': 'Suitcases, handbags and backpacks',
            'Lencería y ropas de baño': 'Lingerie and swimwear',
            'Outdoor': 'Outdoor',
            'Restaurants': 'Restaurants',
            'Moda para niños': "Kids' fashion",
            'Transporte aéreo': 'Air transport',
            'Centros de atención': 'Service centers',
            'Entretenimiento familiar': 'Family entertainment',
            'Peluquería y cuidado personal': 'Hairdressing and personal care',
            'Cine': 'Cinema',
            'Regalos y stationery': 'Gifts and stationery',
            'Jugueterías': 'Toy stores',
            'Tiendas por departamento': 'Department stores',
            'Supermercados': 'Supermarkets',
            'Cajeros automáticos': 'ATMs',
            'Bancos': 'Banks',
            'Casas de cambio': 'Exchange houses',
            'Courier': 'Courier',
            'Servicios': 'Services',
            'Farmacias': 'Pharmacies',
            'Tienda de Mascotas': 'Pet store',
            'Varios': 'Miscellaneous',
            'Autos': 'Cars',
            'Alimentos': 'Food',
            'Celulares e Internet': 'Mobile phones and internet',
            'Decoración': 'Decoration',
            'Artículos para el hogar': 'Household items',
            'Juguería': 'Juice bars',
            'Patio de comidas': 'Food court',
            'Accesorios para celulares': 'Cell phone accessories',
            'Gimnasio': 'Gym',
            'Bicicletas y similares': 'Bicycles and related',
            'Vitaminas y Complementos': 'Vitamins and supplements',
            'Mejoramiento del Hogar': 'Home improvement',
            'Moda urbana': 'Urban fashion',
            'Moda Masculina': "Men's fashion",  # Alternative capitalization
        }

        file_path = options['file']
        
        # Handle relative path from Django project root
        if not os.path.isabs(file_path):
            file_path = os.path.join(settings.BASE_DIR, 'core', 'management', 'commands', file_path)
        
        if not os.path.exists(file_path):
            self.stdout.write(
                self.style.ERROR(f'File not found: {file_path}')
            )
            return

        try:
            # Read Excel file
            self.stdout.write('Reading Excel file...')
            df = pd.read_excel(file_path)
            
            # Print column names to understand the structure
            self.stdout.write(f'Columns found: {list(df.columns)}')
            
            # Assuming columns are: STORE #, STORE NAME, CATEGORY, SUBCATEGORY
            # Adjust column names based on actual Excel structure
            expected_columns = ['STORE #', 'STORE NAME', 'CATEGORY', 'SUBCATEGORY']
            
            # If columns don't match exactly, try to map them
            if not all(col in df.columns for col in expected_columns):
                # Try alternative column mappings
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if 'store' in col_lower and ('#' in col_lower or 'number' in col_lower):
                        column_mapping['store_number'] = col
                    elif 'store' in col_lower and 'name' in col_lower:
                        column_mapping['store_name'] = col
                    elif 'category' in col_lower and 'sub' not in col_lower:
                        column_mapping['category'] = col
                    elif 'subcategory' in col_lower or ('category' in col_lower and 'sub' in col_lower):
                        column_mapping['subcategory'] = col
                
                self.stdout.write(f'Column mapping: {column_mapping}')
            else:
                column_mapping = {
                    'store_number': 'STORE #',
                    'store_name': 'STORE NAME', 
                    'category': 'CATEGORY',
                    'subcategory': 'SUBCATEGORY'
                }

            if options['clear']:
                self.stdout.write('Clearing existing stores...')
                MallStore.objects.all().delete()
                self.stdout.write(self.style.SUCCESS('Existing stores cleared.'))
                store_id_counter = 1
            else:
                # Find the highest existing store ID number to continue sequentially
                existing_stores = MallStore.objects.filter(
                    store_code__startswith='STORE'
                ).order_by('-store_code')
                
                if existing_stores.exists():
                    last_store_code = existing_stores.first().store_code
                    try:
                        # Extract number from STORE001 format
                        last_number = int(last_store_code.replace('STORE', ''))
                        store_id_counter = last_number + 1
                        self.stdout.write(f'Continuing from store ID: STORE{store_id_counter:03d}')
                    except ValueError:
                        store_id_counter = 1
                        self.stdout.write('Could not parse existing store IDs, starting from STORE001')
                else:
                    store_id_counter = 1

            created_count = 0
            updated_count = 0
            error_count = 0

            for index, row in df.iterrows():
                try:
                    # Extract data from row
                    store_number = str(row[column_mapping['store_number']]).strip() if pd.notna(row[column_mapping['store_number']]) else ''
                    store_name = str(row[column_mapping['store_name']]).strip() if pd.notna(row[column_mapping['store_name']]) else ''
                    category_spanish = str(row[column_mapping['category']]).strip() if pd.notna(row[column_mapping['category']]) else ''
                    subcategory_spanish = str(row[column_mapping['subcategory']]).strip() if pd.notna(row[column_mapping['subcategory']]) else ''
                    
                    # Skip empty rows
                    if not store_number or not store_name:
                        continue
                    
                    # Generate sequential store_code in format STORE001, STORE002, etc.
                    store_code = f"STORE{store_id_counter:03d}"
                    
                    # Map categories to English
                    category_english = category_mapping.get(category_spanish, category_spanish)
                    subcategory_english = subcategory_mapping.get(subcategory_spanish, subcategory_spanish)
                    
                    # Create or update store
                    store, created = MallStore.objects.update_or_create(
                        store_code=store_code,
                        defaults={
                            'store_number': store_number,
                            'store_name': store_name,
                            'pattern_characterstic_1': category_english,
                            'pattern_characterstic_2': subcategory_english,
                            'pattern_characterstic_3': '',  # Empty by default
                        }
                    )
                    
                    if created:
                        created_count += 1
                        self.stdout.write(f'Created: {store_code} (#{store_number}) - {store_name}')
                    else:
                        updated_count += 1
                        self.stdout.write(f'Updated: {store_code} (#{store_number}) - {store_name}')
                    
                    # Increment counter for next store
                    store_id_counter += 1
                        
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(f'Error processing row {index + 1}: {str(e)}')
                    )
                    continue

            # Summary
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nImport completed!\n'
                    f'Created: {created_count} stores\n'
                    f'Updated: {updated_count} stores\n'
                    f'Errors: {error_count} rows\n'
                    f'Total processed: {created_count + updated_count}'
                )
            )

            # Show unmapped categories/subcategories for review
            unmapped_categories = set()
            unmapped_subcategories = set()
            
            for index, row in df.iterrows():
                if pd.notna(row[column_mapping['category']]):
                    category_spanish = str(row[column_mapping['category']]).strip()
                    if category_spanish and category_spanish not in category_mapping:
                        unmapped_categories.add(category_spanish)
                
                if pd.notna(row[column_mapping['subcategory']]):
                    subcategory_spanish = str(row[column_mapping['subcategory']]).strip()
                    if subcategory_spanish and subcategory_spanish not in subcategory_mapping:
                        unmapped_subcategories.add(subcategory_spanish)
            
            if unmapped_categories:
                self.stdout.write(
                    self.style.WARNING(f'\nUnmapped categories found: {unmapped_categories}')
                )
            
            if unmapped_subcategories:
                self.stdout.write(
                    self.style.WARNING(f'Unmapped subcategories found: {unmapped_subcategories}')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to process file: {str(e)}')
            ) 